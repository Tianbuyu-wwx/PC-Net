# -*- coding: utf-8 -*-
"""Train and evaluate PC-Net.

    python main.py --name pcnet --epochs 100 --gpu 0
    python main.py --name pcnet --test_only --ckpt outputs/ckpt/pcnet.pt

The training objective is the nine-quantile pinball loss plus the free-energy
term, ``L = pinball + lam_F * F``.  Evaluation reports MAE / RMSE / R^2 /
nQS9 over the 42 station-indicator cells of the eight test origins.
"""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import dataset
import gol
from model import PCNet
from parse import get_args


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def pinball(y_true: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Elementwise pinball loss for ``y_true (N,)`` and ``q (N, 9)``."""
    err = y_true[:, None] - q
    tau = gol.QUANTILES[None, :]
    return np.maximum(err * tau, err * (tau - 1))


def score_frame(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """Attach ``ae``/``se``/``nMAE``/``nQS9`` to a per-step frame."""
    d = df.copy()
    d["ae"] = np.abs(d.y_true - d.q50)
    d["se"] = (d.y_true - d.q50) ** 2
    d["train_sd"] = [stats[(s, v)][1] for s, v in zip(d.section, d.variable)]
    d["nQS9"] = 2.0 * pinball(d.y_true.to_numpy(), d[gol.QCOLS].to_numpy()).mean(1) / d.train_sd
    d["nMAE"] = d.ae / d.train_sd
    return d


def cell_metrics(d: pd.DataFrame) -> pd.DataFrame:
    """One row of metrics per (station, variable)."""
    rows = []
    for (sec, var), g in d.groupby(["section", "variable"], sort=False):
        corr = g.y_true.corr(g.q50)
        rows.append(dict(section=sec, variable=var,
                         MAE=float(g.ae.mean()),
                         RMSE=float(np.sqrt(g.se.mean())),
                         R2=float(corr ** 2),
                         nQS9=float(g.nQS9.mean()),
                         nMAE=float(g.nMAE.mean()),
                         n=int(len(g))))
    return pd.DataFrame(rows)


def report(cells: pd.DataFrame):
    """Print the per-indicator table and the scale-free summary."""
    avg = cells.groupby("variable", sort=False)[["RMSE", "MAE", "R2"]].mean()
    avg = avg.reindex([v for v in gol.VARIABLES if v in avg.index])
    n_sec = cells["section"].nunique() if "section" in cells.columns else 1
    print(f"\nper-indicator average over {n_sec} station{'s' if n_sec > 1 else ''}")
    print(avg.to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"\nnMAE {cells.nMAE.mean():.4f} | nQS9 {cells.nQS9.mean():.4f}")


# --------------------------------------------------------------------------- #
# model helpers
# --------------------------------------------------------------------------- #
def build_model(a) -> PCNet:
    return PCNet(ctx=gol.CTX, hor=gol.HOR, layers=a.layers, tinf=a.tinf, d=a.d,
                 eta=a.eta, pyramid=(a.scales == "pyramid"), coupling=a.coupling,
                 lam_f=a.lam_F, prec=bool(a.prec), readout=a.readout,
                 n_vars=len(gol.VARIABLES), dropout=a.dp,
                 trend_feat=bool(a.trend_feat), cal_fut=bool(a.cal_fut))


def cal_block(anchor_ns: np.ndarray) -> np.ndarray:
    return np.stack([dataset.window_calendar(int(t)) for t in anchor_ns]).astype(np.float32)


def batch_loss(model, X, Y, vid, cal, dev, a, train=True):
    """pinball (or MSE) loss + lam_F * free energy."""
    with torch.autocast("cuda", dtype=torch.bfloat16) if (a.amp and dev == "cuda") \
            else torch.autocast("cpu", enabled=False):
        q, fv, y0 = model(X.to(dev), vid.to(dev), cal.to(dev))
        y = Y.to(dev)
        if a.loss == "pinball":
            tau = torch.tensor(list(gol.QUANTILES), dtype=torch.float32,
                               device=q.device).view(1, 1, 9)
            err = y.unsqueeze(-1) - q
            main = torch.maximum(tau * err, (tau - 1) * err).mean()
        else:
            main = F.mse_loss(y0, y)
        return main + a.lam_F * fv


@torch.no_grad()
def predict(ev: dict, model: PCNet, dev: str, batch: int = 256) -> np.ndarray:
    model.eval()
    out = np.zeros((len(ev["Xn"]), gol.HOR, 9), dtype=np.float64)
    cal = torch.from_numpy(ev["cal"])
    for b in range(0, len(ev["Xn"]), batch):
        q, _, _ = model(torch.from_numpy(ev["Xn"][b:b + batch]).to(dev),
                        torch.from_numpy(ev["var_id"][b:b + batch]).long().to(dev),
                        cal[b:b + batch].to(dev))
        out[b:b + batch] = q.float().cpu().numpy()
    return out


def per_step_frame(ev: dict, q_phys: np.ndarray, name: str) -> pd.DataFrame:
    N, H, _ = q_phys.shape
    rows = {
        "split": np.repeat("test", N * H),
        "section": np.repeat([gol.STATIONS[s] for s in ev["sec_id"]], H),
        "variable": np.repeat([gol.VARIABLES[v] for v in ev["var_id"]], H),
        "anchor": np.repeat(ev["anchor"], H),
        "step": np.tile(np.arange(1, H + 1), N),
        "y_true": ev["Yraw"].reshape(-1).astype(np.float64),
        "method": np.repeat(name, N * H),
    }
    flat = q_phys.reshape(-1, 9)
    for i, c in enumerate(gol.QCOLS):
        rows[c] = flat[:, i]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# train / test
# --------------------------------------------------------------------------- #
def train(a, dev):
    series = dataset.load_series(a.data)
    stats = dataset.train_stats(series)
    pool = dataset.build_train_pool(series, stats)
    valp = dataset.build_val_pool(series, stats)

    Xn, Yn, vid = pool["Xn"], pool["Yn"], pool["var_id"]
    calX = cal_block(pool["at"])
    cval = cal_block(valp["at"])

    N = len(Xn)
    model = build_model(a).to(dev)
    ema = copy.deepcopy(model).eval() if a.ema > 0 else None

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    spe = max(1, N // a.batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=a.lr, total_steps=spe * a.epochs, pct_start=0.05)
    rng = np.random.default_rng(a.seed)
    best_val, best_state = float("inf"), None
    t0 = time.time()

    print(f"PC-Net | {sum(p.numel() for p in model.parameters()):,} parameters | "
          f"{N} train windows | {len(valp['Xn'])} val windows | {dev}")

    for ep in range(1, a.epochs + 1):
        model.train()
        perm = rng.permutation(N)
        tot, nb = 0.0, 0
        for b in range(spe):
            idx = perm[b * a.batch_size:(b + 1) * a.batch_size]
            if len(idx) < 8:
                continue
            loss = batch_loss(model, torch.from_numpy(Xn[idx]), torch.from_numpy(Yn[idx]),
                              torch.from_numpy(vid[idx]).long(),
                              torch.from_numpy(calX[idx]), dev, a)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            if ema is not None:
                with torch.no_grad():
                    for pe, pm in zip(ema.parameters(), model.parameters()):
                        pe.mul_(a.ema).add_(pm, alpha=1 - a.ema)
            tot += float(loss.item()); nb += 1

        ev = ema if ema is not None else model
        ev.eval()
        with torch.no_grad():
            vals = []
            for b in range(0, len(valp["Xn"]), 512):
                vals.append(float(batch_loss(
                    ev, torch.from_numpy(valp["Xn"][b:b + 512]),
                    torch.from_numpy(valp["Yn"][b:b + 512]),
                    torch.from_numpy(valp["var_id"][b:b + 512]).long(),
                    torch.from_numpy(cval[b:b + 512]), dev, a, train=False).item()))
        val = float(np.mean(vals))
        if val < best_val:
            best_val = val
            best_state = {k: v.detach().cpu().clone() for k, v in ev.state_dict().items()}
        if ep % a.log_every == 0 or ep == 1 or ep == a.epochs:
            print(f"ep {ep:3d}/{a.epochs}  train {tot/max(nb,1):.4f}  val {val:.4f}  "
                  f"best {best_val:.4f}  ({(time.time()-t0)/60:.1f} min)")

    out = Path(gol.OUTPUT_PATH) / "ckpt" / f"{a.name}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state_dict=best_state, args=vars(a), best_val=best_val), out)
    (out.parent / f"{a.name}.json").write_text(
        json.dumps(dict(best_val=best_val, params=sum(p.numel() for p in model.parameters()),
                        minutes=(time.time() - t0) / 60,
                        args={k: str(v) for k, v in vars(a).items()}), indent=2),
        encoding="utf-8")
    print(f"saved {out}  best_val={best_val:.4f}")
    return out


def test(a, dev):
    ckpt = Path(a.ckpt) if a.ckpt else Path(gol.OUTPUT_PATH) / "ckpt" / f"{a.name}.pt"
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    saved = ck.get("args", {})
    if a.data is None:                      # reuse the table the checkpoint was trained on
        src = saved if isinstance(saved, dict) else vars(saved)
        a.data = src.get("data")
    series = dataset.load_series(a.data)
    stats = dataset.train_stats(series)
    model = build_model(argparse_namespace(saved, a))
    model.load_state_dict(ck["state_dict"])
    model.to(dev).eval()

    ev = dataset.build_test_set(series, stats)
    q = dataset.denormalise(predict(ev, model, dev), ev)
    df = per_step_frame(ev, q, a.name)
    out = Path(gol.OUTPUT_PATH) / f"{a.name}_per_step.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")

    cells = cell_metrics(score_frame(df, stats))
    print(f"{len(df):,} targets / {len(df)//gol.HOR:,} sequences -> {out}")
    report(cells)
    return out


def argparse_namespace(saved: dict, a):
    """Rebuild the architecture from the checkpoint, keeping the data options."""
    if not isinstance(saved, dict):
        saved = vars(saved)
    for k, v in saved.items():
        if hasattr(a, k) and k not in ("name", "test_only", "ckpt", "data"):
            try:
                setattr(a, k, v)
            except Exception:
                pass
    return a


def main():
    a = get_args()
    if a.gpu >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(a.gpu)
        dev = "cuda"
    else:
        dev = "cpu"
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    print(f"device {dev}")
    try:
        return test(a, dev) if a.test_only else train(a, dev)
    except FileNotFoundError as err:
        raise SystemExit(str(err))


if __name__ == "__main__":
    main()
