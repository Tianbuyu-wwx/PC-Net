# -*- coding: utf-8 -*-
"""Dataset construction for the 4-hourly Taihu water-quality records.

Three of the seven stations are redistributed with this repository as processed
Parquet files (see ``data/``); the complete seven-station table is not.  Any
input must be a long table on a strict 4-hour grid:

    datetime,section,variable,value[,quality,split]

``.parquet``, ``.csv`` and ``.csv.gz`` are all accepted.  Pipeline:
    1. ``load_series``   -> {(station, variable): pandas.Series}
    2. ``train_stats``   -> z-score statistics fitted on 2021-2023 only
    3. ``build_*_pool``  -> NaN-free windows  x[i-ctx:i] -> y[i:i+hor]
    4. ``denormalise``   -> physical units + bounds + monotone quantiles
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import gol


def _ts(x) -> pd.Timestamp:
    return pd.Timestamp(x)


def _read_table(path: Path) -> pd.DataFrame:
    """Read a long table from ``.parquet``, ``.csv`` or ``.csv.gz``."""
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
        df["datetime"] = pd.to_datetime(df["datetime"])
        return df
    return pd.read_csv(path, parse_dates=["datetime"])


def load_series(data_path=None) -> dict:
    """Return ``{(station, variable): pandas.Series}`` on a contiguous 4h grid."""
    path = Path(data_path) if data_path is not None else Path(gol.DATA_PATH)
    if not path.exists():
        raise FileNotFoundError(
            f"dataset not found: {path}\n"
            "Pass --data <file> to use another table; the processed station tables shipped "
            "with this repository are read directly, e.g.\n"
            "    python main.py --data data/taihu_4h_S1_yuexiqiao.parquet --name demo --epochs 5\n"
            "For the complete seven-station table, point DATA_PATH in gol.py to your local copy."
        )
    df = _read_table(path)
    df = df[df.section.isin(gol.STATIONS) & df.variable.isin(gol.VARIABLES)]
    out = {}
    for (sec, var), g in df.groupby(["section", "variable"]):
        s = g.sort_values("datetime").set_index("datetime")["value"].astype(np.float64)
        grid = pd.date_range(s.index.min(), s.index.max(), freq="4h")
        out[(sec, var)] = s.reindex(grid)
    return out


def train_stats(series: dict) -> dict:
    """Per-(station, variable) z-score statistics fitted on the train segment only."""
    st = {}
    for key, s in series.items():
        tr = s[(s.index >= _ts(gol.TRAIN_START)) & (s.index <= _ts(gol.TRAIN_END))].dropna()
        if len(tr) < 30:
            continue
        st[key] = (float(tr.mean()), float(tr.std() + 1e-8))
    return st


# --------------------------------------------------------------------------- #
# calendar covariates (known in advance, no target information)
# --------------------------------------------------------------------------- #
def calendar_covariates(times: pd.DatetimeIndex) -> np.ndarray:
    h, dow, doy = times.hour.values, times.dayofweek.values, times.dayofyear.values
    return np.stack([
        np.sin(2 * np.pi * h / 24.0), np.cos(2 * np.pi * h / 24.0),
        np.sin(2 * np.pi * dow / 7.0), np.cos(2 * np.pi * dow / 7.0),
        np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25),
    ], axis=-1).astype(np.float32)


def window_calendar(anchor_ns: int) -> np.ndarray:
    """(ctx + hor, 6) calendar block for one anchor given as int64 nanoseconds."""
    ts = pd.Timestamp(np.datetime64(int(anchor_ns), "ns"))
    times = pd.date_range(ts - pd.Timedelta(hours=4 * gol.CTX),
                          periods=gol.CTX + gol.HOR, freq="4h")
    return calendar_covariates(times)


# --------------------------------------------------------------------------- #
# window construction
# --------------------------------------------------------------------------- #
def _vec_valid(vals: np.ndarray, cand: np.ndarray) -> np.ndarray:
    cs = np.concatenate([[0], np.cumsum(~np.isnan(vals))])
    return (cs[cand + gol.HOR] - cs[cand - gol.CTX]) == (gol.CTX + gol.HOR)


def _drop_flat(X: np.ndarray, min_keep: int = 60, rel: float = 1e-3) -> np.ndarray:
    """Drop near-constant windows; their targets carry no learnable signal."""
    gstd = np.nanstd(X)
    thr = max(1e-8, rel * (gstd if np.isfinite(gstd) and gstd > 0 else 1.0))
    keep = X.std(axis=1) >= thr
    return keep if keep.sum() >= min_keep else np.ones(len(X), bool)


def _collect(series, stats, picker, drop_flat=False) -> dict:
    Xs, Ys, vid, sid, ats = [], [], [], [], []
    for si, sec in enumerate(gol.STATIONS):
        for vi, var in enumerate(gol.VARIABLES):
            key = (sec, var)
            if key not in series or key not in stats:
                continue
            s = series[key]
            picks = picker(s)
            if len(picks) == 0:
                continue
            vals = s.values.astype(np.float64)
            X = np.stack([vals[i - gol.CTX:i] for i in picks]).astype(np.float32)
            Y = np.stack([vals[i:i + gol.HOR] for i in picks]).astype(np.float32)
            if drop_flat:
                keep = _drop_flat(X)
                picks, X, Y = picks[keep], X[keep], Y[keep]
            mu, sg = stats[key]
            Xs.append(((X - mu) / sg).astype(np.float32))
            Ys.append(((Y - mu) / sg).astype(np.float32))
            vid.append(np.full(len(X), vi, np.int64))
            sid.append(np.full(len(X), si, np.int64))
            ats.append(s.index.values[picks].astype("datetime64[ns]").astype(np.int64))
    if not Xs:
        raise RuntimeError("no windows built - check DATA_PATH in gol.py")
    return dict(Xn=np.concatenate(Xs), Yn=np.concatenate(Ys),
                var_id=np.concatenate(vid), sec_id=np.concatenate(sid),
                at=np.concatenate(ats))


def _train_picker(s):
    dates, vals = s.index.values, s.values.astype(np.float64)
    i_end = int(np.searchsorted(dates, np.datetime64(_ts(gol.TRAIN_END)), side="right"))
    lo, hi = gol.CTX, min(i_end - gol.HOR + 1, len(dates) - gol.HOR)
    if hi <= lo:
        return np.empty(0, np.int64)
    cand = np.arange(lo, hi)
    return cand[_vec_valid(vals, cand)][::gol.STEP]


def _val_picker(s):
    dates, vals = s.index.values, s.values.astype(np.float64)
    lo = int(np.searchsorted(dates, np.datetime64(_ts(gol.TRAIN_END)), side="right"))
    hi = int(np.searchsorted(dates, np.datetime64(_ts(gol.VAL_END)), side="right")) - 1 - gol.HOR + 2
    lo, hi = max(lo, gol.CTX), min(hi, len(dates) - gol.HOR)
    if hi <= lo:
        return np.empty(0, np.int64)
    cand = np.arange(lo, hi)
    return cand[_vec_valid(vals, cand)][::gol.STEP][:64]


def build_train_pool(series=None, stats=None) -> dict:
    """Training windows: non-overlapping stride-64 targets inside 2021-2023."""
    series = load_series() if series is None else series
    stats = train_stats(series) if stats is None else stats
    return _collect(series, stats, _train_picker, drop_flat=True)


def build_val_pool(series=None, stats=None) -> dict:
    """Early-stopping windows taken from 2024 (never used for reported numbers)."""
    series = load_series() if series is None else series
    stats = train_stats(series) if stats is None else stats
    return _collect(series, stats, _val_picker)


def build_test_set(series=None, stats=None) -> dict:
    """Eight-origin test windows, normalised and raw."""
    series = load_series() if series is None else series
    stats = train_stats(series) if stats is None else stats
    rec = {k: [] for k in ["Xraw", "Yraw", "Xn", "Yn", "var_id", "sec_id",
                           "anchor", "mu", "sg", "cal"]}
    for si, sec in enumerate(gol.STATIONS):
        for vi, var in enumerate(gol.VARIABLES):
            key = (sec, var)
            if key not in series or key not in stats:
                continue
            s, mu, sg = series[key], *stats[key]
            dates, vals = s.index.values, s.values.astype(np.float64)
            for a in gol.TEST_ANCHORS:
                i = int(np.searchsorted(dates, np.datetime64(_ts(a)), side="left"))
                if i < gol.CTX or len(vals) - i < gol.HOR:
                    continue
                if np.isnan(vals[i - gol.CTX:i + gol.HOR]).any():
                    continue
                X = vals[i - gol.CTX:i].astype(np.float32)
                Y = vals[i:i + gol.HOR].astype(np.float32)
                rec["Xraw"].append(X); rec["Yraw"].append(Y)
                rec["Xn"].append(((X - mu) / sg).astype(np.float32))
                rec["Yn"].append(((Y - mu) / sg).astype(np.float32))
                rec["var_id"].append(vi); rec["sec_id"].append(si)
                rec["anchor"].append(a)
                rec["mu"].append(mu); rec["sg"].append(sg)
                rec["cal"].append(window_calendar(dates[i]))
    return dict(
        Xraw=np.stack(rec["Xraw"]).astype(np.float32),
        Yraw=np.stack(rec["Yraw"]).astype(np.float32),
        Xn=np.stack(rec["Xn"]).astype(np.float32),
        Yn=np.stack(rec["Yn"]).astype(np.float32),
        var_id=np.array(rec["var_id"], np.int64),
        sec_id=np.array(rec["sec_id"], np.int64),
        anchor=np.array(rec["anchor"], dtype="U10"),
        mu=np.array(rec["mu"], np.float64),
        sg=np.array(rec["sg"], np.float64),
        cal=np.stack(rec["cal"]).astype(np.float32))


# --------------------------------------------------------------------------- #
# post-processing
# --------------------------------------------------------------------------- #
def monoticise(q: np.ndarray) -> np.ndarray:
    """Remove the quantile crossings that would corrupt the pinball score.

    The median is left untouched, so MAE and RMSE are unaffected by this step.
    """
    q = np.array(q, dtype=np.float64, copy=True)
    q[..., 0] = np.minimum(q[..., 0], q[..., 4])
    q[..., 8] = np.maximum(q[..., 8], q[..., 4])
    q[..., 1] = np.minimum(q[..., 1], q[..., 2])
    q[..., 7] = np.maximum(q[..., 7], q[..., 6])
    return q


def denormalise(q_z: np.ndarray, ev: dict) -> np.ndarray:
    """z-space quantiles -> physical units, clipped to bounds and monotone in tau."""
    q = q_z * ev["sg"][:, None, None] + ev["mu"][:, None, None]
    for vi, var in enumerate(gol.VARIABLES):
        m = ev["var_id"] == vi
        lo, hi = gol.PHYS_BOUNDS[var]
        q[m] = np.clip(q[m], lo, hi)
    return monoticise(q)
