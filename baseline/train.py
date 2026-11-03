# -*- coding: utf-8 -*-
"""Train and evaluate the six deep baselines compared against PC-Net.

The script reuses ``dataset.py`` of the main model, so every baseline sees
exactly the same 256-step contexts, the same 64-step targets, the same
chronological split and the same z-score statistics as PC-Net.  Each baseline
keeps its own architecture; the nine-quantile head, the pinball loss and the
42-cell evaluation are shared.

``DLinear`` is a point forecaster, so it is trained with the squared error and
only its ``q50`` column is populated — this mirrors the paper, where the
probabilistic metrics of DLinear are reported as undefined.

Usage
-----
    python baseline/train.py --model patchtst --data data/taihu_4h_S1_yuexiqiao.parquet
    python baseline/train.py --model all      --data data/taihu_4h_S1_yuexiqiao.parquet
    python baseline/train.py --model lstm --test_only --ckpt outputs/ckpt/baseline/lstm.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import dataset  # noqa: E402  (the shared window / statistics code)
import gol  # noqa: E402
from baseline.dlinear import DLinear  # noqa: E402
from baseline.itransformer import ITransformerModel  # noqa: E402
from baseline.lstm import LSTMModel  # noqa: E402
from baseline.nbeats import NBeatsModel  # noqa: E402
from baseline.patchtst import PatchTSTModel  # noqa: E402
from baseline.transformer import TransformerModel  # noqa: E402

MODELS = {
    "lstm": (LSTMModel, dict(hidden=128, layers=2, out=gol.HOR)),
    "dlinear": (DLinear, dict(ctx=gol.CTX, hor=gol.HOR)),
    "nbeats": (NBeatsModel, dict(ctx=gol.CTX, horizon=gol.HOR, stacks=8, layers=4, hidden=256)),
    "transformer": (TransformerModel, dict(d_model=128, nhead=8, e_layers=2, d_layers=1, out=gol.HOR)),
    "patchtst": (PatchTSTModel, dict(patch=16, stride=8, d_model=128, nhead=8, layers=3,
                                     out=gol.HOR, drop=0.2, ctx=gol.CTX)),
    "itransformer": (ITransformerModel, dict(ctx=gol.CTX, d_model=256, nhead=8, layers=2, out=gol.HOR)),
}

POINT_ONLY = {"dlinear"}          # no quantile head, trained with the squared error
NEEDS_2D = {"nbeats", "dlinear"}  # (B, T) instead of (B, T, 1)
NEEDS_VT = {"itransformer"}       # (B, V, T) with V = 1


def adapt(name: str, x: torch.Tensor) -> torch.Tensor:
    """Reshape a ``(B, T)`` window into what the given baseline expects."""
    if name in NEEDS_2D:
        return x
    if name in NEEDS_VT:
        return x.unsqueeze(1)
    return x.unsqueeze(-1)


def pinball(y: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    tau = torch.arange(1, 10, device=q.device, dtype=q.dtype) / 10
    err = y[:, :, None] - q
    return torch.maximum(err * tau, err * (tau - 1)).mean()


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
def train_model(name: str, pool: dict, epochs: int, batch: int, lr: float,
                wd: float, dev: str, seed: int, max_samples: int | None = None,
                log_every: int = 5) -> nn.Module:
    torch.manual_seed(seed)
    np.random.seed(seed)
    X = torch.tensor(pool["Xn"], dtype=torch.float32)
    Y = torch.tensor(pool["Yn"], dtype=torch.float32)
    if max_samples and len(X) > max_samples:
        idx = torch.randperm(len(X))[:max_samples]
        X, Y = X[idx], Y[idx]
    print(f"[{name}] train windows: {tuple(X.shape)} | parameters: "
          f"{sum(p.numel() for p in MODELS[name][0](**MODELS[name][1]).parameters()):,}")

    model = MODELS[name][0](**MODELS[name][1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    steps_per_epoch = max(1, len(X) // batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps_per_epoch * epochs)

    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(X))
        total, seen = 0.0, 0
        for i in range(0, len(X), batch):
            ids = perm[i:i + batch]
            xb, yb = X[ids].to(dev), Y[ids].to(dev)
            out = model(adapt(name, xb))
            loss = nn.functional.mse_loss(out, yb) if name in POINT_ONLY else pinball(yb, out)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total += float(loss) * len(ids)
            seen += len(ids)
        if ep % log_every == 0 or ep == epochs:
            print(f"[{name}] epoch {ep}/{epochs}  loss {total / max(seen, 1):.4f}")
    return model


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def predict(model: nn.Module, name: str, xn: np.ndarray, dev: str, batch: int = 256) -> np.ndarray:
    model.eval()
    outs = []
    for i in range(0, len(xn), batch):
        xb = torch.tensor(xn[i:i + batch], dtype=torch.float32, device=dev)
        outs.append(model(adapt(name, xb)).float().cpu().numpy())
    return np.concatenate(outs, 0)


def to_frame(name: str, q_phys: np.ndarray, ev: dict) -> pd.DataFrame:
    """Flatten ``(N, H, 9)`` predictions into the per-step layout used by the paper."""
    n, hor = q_phys.shape[0], q_phys.shape[1]
    frame = pd.DataFrame({
        "split": "test",
        "section": np.repeat(ev["sec_id"], hor),
        "variable": np.repeat(ev["var_id"], hor),
        "anchor": np.repeat(ev["anchor"], hor),
        "step": np.tile(np.arange(1, hor + 1), n),
        "y_true": ev["Yraw"].reshape(-1).astype(np.float64),
    })
    frame["section"] = frame["section"].map(lambda i: gol.STATIONS[int(i)])
    frame["variable"] = frame["variable"].map(lambda i: gol.VARIABLES[int(i)])
    flat = q_phys.reshape(-1, q_phys.shape[-1])
    for j, col in enumerate(gol.QCOLS):
        frame[col] = flat[:, j]
    frame["method"] = name
    return frame


def score(name: str, frame: pd.DataFrame, stats: dict) -> pd.DataFrame:
    d = frame.copy()
    d["ae"] = np.abs(d.y_true - d.q50)
    d["se"] = (d.y_true - d.q50) ** 2
    d["train_sd"] = [stats[(s, v)][1] for s, v in zip(d.section, d.variable)]
    d["nMAE"] = d.ae / d.train_sd
    q = d[gol.QCOLS].to_numpy()
    if name in POINT_ONLY or np.isnan(q).any():
        d["nQS9"] = np.nan
    else:
        tau = gol.QUANTILES[None, :]
        err = d.y_true.to_numpy()[:, None] - q
        d["nQS9"] = 2.0 * np.maximum(err * tau, err * (tau - 1)).mean(1) / d.train_sd
    rows = []
    for (sec, var), g in d.groupby(["section", "variable"], sort=False):
        rows.append(dict(section=sec, variable=var,
                         MAE=float(g.ae.mean()),
                         RMSE=float(np.sqrt(g.se.mean())),
                         R2=float(g.y_true.corr(g.q50) ** 2),
                         nQS9=float(g.nQS9.mean()),
                         nMAE=float(g.nMAE.mean()),
                         n=int(len(g))))
    return d, pd.DataFrame(rows)


def report(name: str, cells: pd.DataFrame) -> None:
    avg = cells.groupby("variable", sort=False)[["RMSE", "MAE", "R2"]].mean()
    avg = avg.reindex([v for v in gol.VARIABLES if v in avg.index])
    print(f"\n[{name}] per-indicator average over {cells.section.nunique()} station(s)")
    print(avg.to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"[{name}] nMAE {cells.nMAE.mean():.4f} | nQS9 {cells.nQS9.mean():.4f}")


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def get_args():
    p = argparse.ArgumentParser("Taifu baseline harness (same protocol as PC-Net)")
    p.add_argument("--model", default="lstm", help="one of " + ", ".join(MODELS) + ", or 'all'")
    p.add_argument("--data", default=None, help="processed table; defaults to gol.DATA_PATH")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--max_train_samples", type=int, default=None, help="cap the training windows")
    p.add_argument("--seed", type=int, default=gol.SEED)
    p.add_argument("--gpu", type=int, default=0, help="-1 for CPU")
    p.add_argument("--out", default="outputs/baseline", help="directory for checkpoints and CSV")
    p.add_argument("--test_only", action="store_true", help="evaluate --ckpt without training")
    p.add_argument("--ckpt", default=None, help="checkpoint for --test_only")
    return p.parse_args()


def main():
    a = get_args()
    dev = f"cuda:{a.gpu}" if (a.gpu >= 0 and torch.cuda.is_available()) else "cpu"
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    out_dir = Path(a.out)
    (out_dir / "ckpt").mkdir(parents=True, exist_ok=True)
    print(f"device {dev}")

    series = dataset.load_series(a.data)
    stats = dataset.train_stats(series)
    names = list(MODELS) if a.model == "all" else [a.model]
    pool = None if a.test_only else dataset.build_train_pool(series, stats)
    ev = dataset.build_test_set(series, stats)

    summary = {}
    for name in names:
        if a.test_only:
            ckpt = Path(a.ckpt) if a.ckpt and len(names) == 1 else out_dir / "ckpt" / f"{name}.pt"
            model = MODELS[name][0](**MODELS[name][1]).to(dev)
            model.load_state_dict(torch.load(ckpt, map_location=dev))
            print(f"[{name}] loaded {ckpt}")
        else:
            model = train_model(name, pool, a.epochs, a.batch_size, a.lr, a.weight_decay,
                                dev, a.seed, a.max_train_samples)
            torch.save(model.state_dict(), out_dir / "ckpt" / f"{name}.pt")

        q_z = predict(model, name, ev["Xn"], dev)
        if name in POINT_ONLY:                     # (N, H) -> (N, H, 9) with only q50
            q9 = np.full((q_z.shape[0], q_z.shape[1], 9), np.nan, dtype=np.float64)
            q9[..., 4] = q_z
            q_z = q9
        q_phys = q_z * ev["sg"][:, None, None] + ev["mu"][:, None, None]
        frame, cells = score(name, to_frame(name, q_phys, ev), stats)
        report(name, cells)
        frame.to_csv(out_dir / f"{name}_per_step.csv", index=False)
        cells.to_csv(out_dir / f"{name}_cells.csv", index=False)
        summary[name] = dict(MAE=float(cells.MAE.mean()), RMSE=float(cells.RMSE.mean()),
                             R2=float(cells.R2.mean()), nMAE=float(cells.nMAE.mean()),
                             nQS9=None if cells.nQS9.isna().all() else float(cells.nQS9.mean()))

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n" + json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
