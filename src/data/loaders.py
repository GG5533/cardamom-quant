"""Real-data facade — the swap-in point that replaces synthetic.py.

Public API (what run.py calls):

    load_spot()      -> daily Spices Board auction aggregate
    load_futures()   -> MCX cardamom continuous front-month
    load_rain()      -> Idukki rainfall + anomaly
    build_market_dataset() -> single aligned frame for feature engineering

Alignment policy (explicit, because this is where leakage hides):
  * master calendar = union of spot and futures dates (weekdays only);
  * spot is forward-filled for basis computation, but staleness is carried
    in ``spot_staleness_days`` — the feature layer can (and should) mask
    stale-basis rows rather than trust a silent ffill;
  * basis columns exist only where BOTH a futures close and a spot print
    within ``max_spot_staleness`` days exist. Everything else is NaN, and
    NaN means "not knowable that day", never "zero".
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .imd_rainfall import IMDRainfallLoader
from .mcx_bhavcopy import MCXBhavcopyLoader
from .spices_board import SpicesBoardLoader

logger = logging.getLogger(__name__)

__all__ = [
    "SpicesBoardLoader",
    "MCXBhavcopyLoader",
    "IMDRainfallLoader",
    "load_spot",
    "load_futures",
    "load_rain",
    "build_market_dataset",
]


def load_spot(refresh: bool = False) -> pd.DataFrame:
    return SpicesBoardLoader().load(refresh=refresh)


def load_futures(refresh: bool = False) -> pd.DataFrame:
    return MCXBhavcopyLoader().load(refresh=refresh)


def load_rain(refresh: bool = False) -> pd.DataFrame:
    return IMDRainfallLoader().load(refresh=refresh)


def build_market_dataset(
    spot: pd.DataFrame | None = None,
    futures: pd.DataFrame | None = None,
    rain: pd.DataFrame | None = None,
    max_spot_staleness: int = 3,
) -> pd.DataFrame:
    """Join the three feeds on a master calendar, compute basis honestly."""
    spot = load_spot() if spot is None else spot
    rain = load_rain() if rain is None else rain
    if futures is None:
        try:
            futures = load_futures()
        except FileNotFoundError:
            logger.warning("no MCX files yet — building spot-only dataset")
            futures = None

    idx = spot.index
    if futures is not None:
        idx = idx.union(futures.index)
    idx = idx[idx.dayofweek < 6]  # auctions run Mon–Sat; drop Sundays only
    df = pd.DataFrame(index=idx.sort_values())
    df.index.name = "date"

    # --- spot with explicit staleness ------------------------------------
    df = df.join(spot, how="left")
    last_print = df["spot_avg"].notna()
    df["spot_staleness_days"] = (~last_print).groupby(last_print.cumsum()).cumsum()
    df[["spot_avg", "spot_max", "qty_arrived", "qty_sold"]] = df[
        ["spot_avg", "spot_max", "qty_arrived", "qty_sold"]
    ].ffill()

    # --- futures + basis ---------------------------------------------------
    if futures is not None:
        df = df.join(futures, how="left")
        ok = (
            df["fut_close"].notna()
            & df["spot_avg"].notna()
            & (df["spot_staleness_days"] <= max_spot_staleness)
        )
        df["basis"] = np.where(ok, df["fut_close"] - df["spot_avg"], np.nan)
        df["basis_pct"] = np.where(ok, df["basis"] / df["spot_avg"], np.nan)
    else:
        for c in MCXBhavcopyLoader.SCHEMA:
            df[c] = np.nan
        df["basis"] = np.nan
        df["basis_pct"] = np.nan

    # --- rainfall (calendar-daily; reindex, never ffill a flow variable) ---
    df = df.join(rain[["rain_mm", "rain_climatology", "rain_anomaly"]], how="left")

    return df
