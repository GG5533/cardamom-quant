"""Leakage-safe feature engineering.

Contract: every feature value dated t uses only information available at the
close of t-1 (except calendar features, which are ex-ante knowable). The
target is the sign of the FORWARD 5-day spot return. A unit test mutates
day-t inputs and asserts day-t features are unchanged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HORIZON = 5  # forward return horizon, days

CORE_FEATURE_COLS = [
    "doy_sin", "doy_cos", "lean_season",
    "mom_5", "mom_10", "mom_21", "mom_63",
    "vol_21",
    "basis_pct", "basis_chg",
    "rain_anom_30", "rain_anom_90",
]


def build_features(
    market: pd.DataFrame,
    alt: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """market: aligned frame from build_market_dataset (or synthetic clone).

    Returns (X, y): X strictly lagged features, y forward 5d up/down label.
    Rows where the label is undefined (tail) are dropped; feature NaNs are
    left in place — models that can't handle them get an imputing pipeline.
    """
    px = market["spot_avg"]
    df = pd.DataFrame(index=market.index)
    df.index.name = "date"

    # --- deterministic calendar (ex-ante, no lag needed) -------------------
    doy = df.index.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    df["lean_season"] = df.index.month.isin([3, 4, 5, 6, 7]).astype(float)

    # --- price-derived, all shifted 1 --------------------------------------
    ret1 = px.pct_change()
    for k in (5, 10, 21, 63):
        df[f"mom_{k}"] = px.pct_change(k).shift(1)
    df["vol_21"] = ret1.rolling(21, min_periods=15).std().shift(1)

    # --- basis (only defined post-relaunch; NaN elsewhere by construction) -
    if "basis_pct" in market:
        df["basis_pct"] = market["basis_pct"].shift(1)
        df["basis_chg"] = market["basis_pct"].diff(5).shift(1)
    else:
        df["basis_pct"] = np.nan
        df["basis_chg"] = np.nan

    # --- weather: lagged aggregates of the anomaly -------------------------
    if "rain_anomaly" in market:
        ra = market["rain_anomaly"]
        df["rain_anom_30"] = ra.rolling(30, min_periods=15).mean().shift(1)
        df["rain_anom_90"] = ra.rolling(90, min_periods=45).mean().shift(1)
    else:
        df["rain_anom_30"] = np.nan
        df["rain_anom_90"] = np.nan

    cols = list(CORE_FEATURE_COLS)
    if alt is not None:
        overlap = set(df.columns) & set(alt.columns)
        if overlap:
            raise ValueError(f"alt features collide with core: {sorted(overlap)}")
        df = df.join(alt, how="left")
        cols += [c for c in alt.columns]

    # --- target -------------------------------------------------------------
    fwd = px.shift(-HORIZON) / px - 1.0
    y = (fwd > 0).astype(float)
    valid = fwd.notna() & px.notna()

    return df.loc[valid, cols], y[valid].rename("y_up")


def forward_returns(market: pd.DataFrame, horizon: int = HORIZON) -> pd.Series:
    """The realized forward return the backtest trades against."""
    px = market["spot_avg"]
    return (px.shift(-horizon) / px - 1.0).rename(f"fwd_{horizon}d")
