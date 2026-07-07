"""Scorecard metrics — reported honestly, baseline included."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def classification_metrics(y_true: pd.Series, proba_up: pd.Series) -> dict:
    y = y_true.to_numpy()
    p = proba_up.to_numpy()
    pred = (p > 0.5).astype(float)
    base_rate = max(y.mean(), 1 - y.mean())  # majority-class hit rate
    out = {
        "n": int(len(y)),
        "hit_rate": float((pred == y).mean()),
        "base_rate": float(base_rate),
        "hit_vs_base_pts": float(((pred == y).mean() - base_rate) * 100),
    }
    try:
        out["auc"] = float(roc_auc_score(y, p))
    except ValueError:  # single-class fold
        out["auc"] = float("nan")
    return out


def backtest_metrics(net_ret: pd.Series) -> dict:
    r = net_ret.dropna()
    if len(r) < 30:
        return {"sharpe": float("nan"), "max_dd": float("nan"), "total_ret": float("nan")}
    ann = np.sqrt(252)
    sharpe = float(r.mean() / r.std() * ann) if r.std() > 0 else float("nan")
    equity = (1 + r).cumprod()
    dd = (equity / equity.cummax() - 1.0).min()
    return {
        "sharpe": sharpe,
        "max_dd": float(dd),
        "total_ret": float(equity.iloc[-1] - 1.0),
        "ann_vol": float(r.std() * ann),
    }


def format_scorecard(rows: dict[str, dict]) -> str:
    cols = ["hit_rate", "hit_vs_base_pts", "auc", "sharpe", "max_dd", "total_ret"]
    lines = [f"{'model':<20}" + "".join(f"{c:>18}" for c in cols)]
    for name, m in rows.items():
        vals = []
        for c in cols:
            v = m.get(c, float("nan"))
            vals.append(f"{v:>18.3f}" if v == v else f"{'—':>18}")
        lines.append(f"{name:<20}" + "".join(vals))
    return "\n".join(lines)
