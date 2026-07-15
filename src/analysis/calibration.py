"""Probability calibration + regime-conditional performance.

Calibration: a 0.65 forecast should be right ~65% of the time. Conviction
sizing (2p-1) is only defensible if the probabilities mean something.

Regimes: a signal that only works in La Niña, or only in the lean season,
is a different (and riskier) product than an all-weather one. This module
reports performance sliced by ENSO phase and by crop season so the claim
"the model works" comes with its conditions attached.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


# --------------------------------------------------------------- calibration
def isotonic_calibrator(p_holdout: pd.Series, y_holdout: pd.Series) -> IsotonicRegression:
    """Fit an isotonic map raw p -> calibrated p on a held-out slice.

    The slice must be data the base model never trained on and must be
    purged from the training window by >= the label horizon. Fit one per
    walk-forward fold; apply only to that fold's test probabilities.
    (Equivalent to CalibratedClassifierCV(cv="prefit", method="isotonic")
    without the deprecated prefit API.)
    """
    df = pd.DataFrame({"p": p_holdout, "y": y_holdout}).dropna()
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(df["p"].to_numpy(), df["y"].to_numpy())
    return iso



def calibration_table(y_true: pd.Series, proba_up: pd.Series, n_bins: int = 8) -> pd.DataFrame:
    df = pd.DataFrame({"y": y_true, "p": proba_up}).dropna()
    # quantile bins so each row has comparable sample size
    df["bin"] = pd.qcut(df["p"], q=n_bins, duplicates="drop")
    g = df.groupby("bin", observed=True)
    out = pd.DataFrame(
        {
            "p_mean": g["p"].mean(),
            "y_rate": g["y"].mean(),
            "n": g.size(),
        }
    )
    out["gap"] = out["y_rate"] - out["p_mean"]
    return out


def brier_score(y_true: pd.Series, proba_up: pd.Series) -> float:
    df = pd.DataFrame({"y": y_true, "p": proba_up}).dropna()
    return float(((df["p"] - df["y"]) ** 2).mean())


def calibration_summary(y_true: pd.Series, proba_up: pd.Series) -> dict:
    tab = calibration_table(y_true, proba_up)
    clim = y_true.mean()
    return {
        "brier": brier_score(y_true, proba_up),
        "brier_climatology": float(((clim - y_true) ** 2).mean()),
        "max_abs_gap": float(tab["gap"].abs().max()),
        "n_bins": len(tab),
    }


# ------------------------------------------------------------------- regimes
def enso_phase(oni: pd.Series) -> pd.Series:
    """'elnino' (>= +0.5), 'lanina' (<= -0.5), else 'neutral'."""
    return pd.Series(
        np.select([oni >= 0.5, oni <= -0.5], ["elnino", "lanina"], "neutral"),
        index=oni.index,
        name="enso",
    )


def regime_performance(
    net_ret: pd.Series,
    y_true: pd.Series,
    proba_up: pd.Series,
    regimes: pd.DataFrame,
) -> pd.DataFrame:
    """regimes: frame of categorical columns on the same calendar
    (e.g. enso phase, lean_season). Returns per-regime Sharpe/hit/n."""
    df = pd.DataFrame({"ret": net_ret, "y": y_true, "p": proba_up}).join(regimes)
    df = df.dropna(subset=["ret", "y", "p"])
    rows = []
    for col in regimes.columns:
        for level, grp in df.groupby(col, observed=True):
            if len(grp) < 30:
                continue
            sd = grp["ret"].std()
            rows.append(
                {
                    "regime": f"{col}={level}",
                    "n_days": len(grp),
                    "sharpe": grp["ret"].mean() / sd * np.sqrt(252) if sd > 0 else np.nan,
                    "hit_rate": ((grp["p"] > 0.5) == (grp["y"] > 0.5)).mean(),
                    "avg_daily_ret_bps": grp["ret"].mean() * 1e4,
                }
            )
    return pd.DataFrame(rows).set_index("regime").sort_values("sharpe", ascending=False)
