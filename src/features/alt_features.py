"""Alternative-data feature block: microstructure + calendars + climate + macro.

The headline signal here costs nothing — it was already in the Spices Board
data and nobody looks at it:

  AUCTION MICROSTRUCTURE. Every auction day reports quantity ARRIVED and
  quantity SOLD. Their ratio is a demand-tension gauge (buyers absorbing
  everything vs walking away), and arrivals against their seasonal normal is
  a real-time SUPPLY SURPRISE — physical crop information that price history
  cannot contain. Both are computed strictly from same-day-or-earlier data.

Everything is assembled leakage-safe:
  * microstructure/rolling features use only trailing windows (shifted 1 day
    so a day's own print predicts the NEXT day, matching engineering.py);
  * calendars are ex-ante knowable by definition;
  * climate indices and Comtrade arrive publication-lagged from their loaders.

`build_alt_features()` returns one frame on the spot calendar; append its
columns to FEATURE_COLS in features/engineering.py and re-run walk-forward.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..data.calendars import add_calendar_features
from ..data.climate_indices import to_daily_features as climate_daily

logger = logging.getLogger(__name__)

ALT_FEATURE_COLS = [
    # microstructure
    "tension", "tension_z_63", "arrivals_surprise",
    # demand calendars
    "days_to_ramadan", "ramadan_stocking", "ramadan_proximity",
    "days_to_diwali", "diwali_window",
    # climate teleconnections
    "oni", "oni_elnino", "oni_lanina", "dmi",
    # macro / cross-market (optional, NaN if loaders not run)
    "usdinr", "inr_mom_21", "inr_z_63",
    "gtm_exp_kg_yoy", "gtm_supply_shock",
]


# ------------------------------------------------------------ microstructure
def auction_microstructure(spot_daily: pd.DataFrame) -> pd.DataFrame:
    """Demand tension + supply surprise from qty columns. Lag-safe (shifted).

    tension            sold/arrived, EW-smoothed 5d, shifted 1d
    tension_z_63       tension vs its trailing 63d distribution
    arrivals_surprise  log(arrivals / trailing-1y same-season normal), shifted

    'Seasonal normal' is a trailing 30-day-of-year window mean over the past
    ~1y of history — computed with only past data (rolling, then shift), so
    no climatology-style full-sample leak.
    """
    df = pd.DataFrame(index=spot_daily.index)
    df.index.name = "date"

    ratio = (spot_daily["qty_sold"] / spot_daily["qty_arrived"]).clip(0, 1.5)
    tension = ratio.ewm(span=5, min_periods=3).mean()
    df["tension"] = tension.shift(1)

    mu = tension.rolling(63, min_periods=40).mean()
    sd = tension.rolling(63, min_periods=40).std()
    df["tension_z_63"] = ((tension - mu) / sd).shift(1)

    # trailing seasonal normal for arrivals: mean of arrivals over the past
    # 252 trading-ish days, restricted implicitly by the annual cycle via a
    # centred-free trailing window; simple, causal, good enough as v1
    arr = spot_daily["qty_arrived"].astype(float)
    trailing_normal = arr.rolling(252, min_periods=120).mean()
    df["arrivals_surprise"] = np.log(arr / trailing_normal).shift(1)

    return df


# ------------------------------------------------------------------ assembly
def build_alt_features(
    spot_daily: pd.DataFrame,
    oni: pd.Series | None = None,
    dmi: pd.Series | None = None,
    usdinr_features: pd.DataFrame | None = None,
    gtm_features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One aligned alt-feature frame on the spot calendar.

    Optional blocks default to NaN columns so the pipeline runs with any
    subset of feeds wired — missingness is explicit, never silent zeros.
    """
    idx = spot_daily.index
    parts = [auction_microstructure(spot_daily), add_calendar_features(idx)]

    if oni is not None:
        parts.append(climate_daily(idx, oni, dmi))
    if usdinr_features is not None:
        parts.append(usdinr_features.reindex(idx))
    if gtm_features is not None:
        parts.append(gtm_features.reindex(idx))

    out = pd.concat(parts, axis=1)
    for col in ALT_FEATURE_COLS:  # guarantee stable schema
        if col not in out.columns:
            out[col] = np.nan
    dupes = out.columns[out.columns.duplicated()]
    if len(dupes):
        raise ValueError(f"alt_features: duplicated columns {list(dupes)}")
    return out[ALT_FEATURE_COLS]
