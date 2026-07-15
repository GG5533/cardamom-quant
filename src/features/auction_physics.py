"""Auction-physics features — signal mined from columns the pipeline never
touched, plus a long-memory regime dial.

Three ideas, each with a mechanism (not a data-dredge):

* **Bid dispersion** (`spot_max / spot_avg - 1`). Auction theory: the premium
  of the best lot over the session average mixes quality dispersion with
  bidder competition. Quality moves slowly with the crop calendar; a fast
  widening against its own 63d norm means an eager marginal bidder — demand
  pressure the average price hasn't printed yet.

* **Inventory overhang** (Kaldor–Working storage theory). Cardamom is stored
  across the crop year; prices spike and revert asymmetrically when stocks
  are thin. Proxy: cumulative arrivals since the crop-year start (Aug-1)
  versus the same point of PAST crop years only — an expanding, strictly
  out-of-sample seasonal norm.

* **Rolling Hurst exponent** (R/S estimator, ported from the Jarvis world
  model). H > 0.5 = trending tape, H < 0.5 = mean-reverting tape; gives the
  tree model a regime dial instead of making it infer one from momentum.

Leakage contract: same as engineering.py — everything shifted one day; the
overhang norm uses only completed prior crop years.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PHYSICS_FEATURE_COLS = [
    "bid_dispersion", "dispersion_z_63", "inv_overhang", "hurst_63",
]

_CROP_YEAR_START_MONTH = 8   # harvest ramps from August
_OVERHANG_BUCKET_DAYS = 10   # day-of-crop-year buckets for the norm


def hurst_rs(series: np.ndarray) -> float:
    """R/S Hurst exponent estimate (three-scale log-log slope)."""
    n = len(series)
    if n < 20:
        return 0.5
    lags, rs_vals = [], []
    for sub_n in (n // 4, n // 2, n):
        if sub_n < 8:
            continue
        chunk = series[:sub_n]
        dev = np.cumsum(chunk - chunk.mean())
        r = dev.max() - dev.min()
        s = chunk.std(ddof=1)
        if s > 1e-12 and r > 0:
            lags.append(np.log(sub_n))
            rs_vals.append(np.log(r / s))
    if len(lags) < 2:
        return 0.5
    return float(np.clip(np.polyfit(lags, rs_vals, 1)[0], 0.01, 0.99))


def _crop_year(idx: pd.DatetimeIndex) -> np.ndarray:
    """Crop-year label: Aug-2019..Jul-2020 -> 2019."""
    return idx.year - (idx.month < _CROP_YEAR_START_MONTH)


def _day_in_crop_year(idx: pd.DatetimeIndex) -> np.ndarray:
    start = pd.to_datetime(
        {"year": _crop_year(idx), "month": _CROP_YEAR_START_MONTH, "day": 1}
    )
    return (idx - pd.DatetimeIndex(start)).days.to_numpy()


def inventory_overhang(qty_arrived: pd.Series) -> pd.Series:
    """log(crop-year-to-date arrivals / past-years' norm at the same point).

    Positive = supply running ahead of history (stocks building, price
    pressure down); negative = thin stocks (spike-prone regime). The norm
    for crop year Y averages ONLY crop years < Y — first observed year is
    NaN by construction, never filled.
    """
    idx = qty_arrived.index
    cy = pd.Series(_crop_year(idx), index=idx)
    bucket = pd.Series(_day_in_crop_year(idx) // _OVERHANG_BUCKET_DAYS, index=idx)
    cum = qty_arrived.groupby(cy).cumsum()

    # last cumulative value per (crop year, bucket) -> past-only expanding mean
    per = pd.DataFrame({"cy": cy, "bucket": bucket, "cum": cum})
    ref = per.groupby(["cy", "bucket"])["cum"].last().reset_index()
    ref = ref.sort_values(["bucket", "cy"])
    ref["norm"] = (
        ref.groupby("bucket")["cum"].transform(lambda s: s.expanding().mean().shift())
    )
    norm = per.merge(ref[["cy", "bucket", "norm"]], on=["cy", "bucket"], how="left")
    norm = pd.Series(norm["norm"].to_numpy(), index=idx)

    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.log(cum / norm)
    return pd.Series(out, index=idx).replace([np.inf, -np.inf], np.nan)


def build_physics_features(market: pd.DataFrame) -> pd.DataFrame:
    """Assemble the block on the market calendar; all columns lag-safe."""
    px = market["spot_avg"]
    df = pd.DataFrame(index=market.index)
    df.index.name = "date"

    disp = market["spot_max"] / px - 1.0
    mu = disp.rolling(63, min_periods=40).mean()
    sd = disp.rolling(63, min_periods=40).std()
    df["bid_dispersion"] = disp.shift(1)
    df["dispersion_z_63"] = ((disp - mu) / sd).shift(1)

    df["inv_overhang"] = inventory_overhang(market["qty_arrived"]).shift(1)

    logret = np.log(px).diff()
    df["hurst_63"] = (
        logret.rolling(63, min_periods=63)
        .apply(lambda w: hurst_rs(w.to_numpy()), raw=False)
        .shift(1)
    )
    return df[PHYSICS_FEATURE_COLS]
