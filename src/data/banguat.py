"""Guatemala cardamom export volume — Banco de Guatemala primary source.

UN Comtrade (the mirror) now requires a subscription key, so this loader
goes to the origin instead: Banguat's "Volumen de las Exportaciones (FOB)
de Productos Agrícolas Seleccionados" series, ANNUAL 1994→present, in
millions of quintales. Guatemala is the world's #1 cardamom exporter; its
crop-year supply level is a slow state variable, so annual frequency is
coarse but mechanistically honest — the 2024-25 crop collapse (−45% then
−42% y/y) is exactly what this series carries.

Provenance: data/raw/banguat/volumen_1994_2025.xlsx (pristine download,
banguat.gob.gt → Sector Externo) plus a committed CSV extraction the
parser reads (xlsx parsing needs openpyxl, which is optional here).
Refresh via scripts/refresh_banguat.py wherever openpyxl is available.

Leakage contract: the figure for calendar year Y is treated as published
on 01-Apr of Y+1 (Banguat closes preliminary annual trade data in Q1; the
p/ marker on the newest year says preliminary). Features dated t use only
years whose publication date is ≤ t. A unit test mutates year Y and
asserts every feature before 01-Apr-(Y+1) is unchanged.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config

GTM_FEATURE_COLS = ["gtm_vol_yoy", "gtm_vol_deficit"]
QUINTAL_KG = 45.359237
PUBLICATION = {"month": 4, "day": 1}   # year Y usable from 01-Apr-(Y+1)
CSV_PATH = config.RAW_DIR / "banguat" / "volumen_agricolas_annual.csv"


def load_annual_kg(path: Path = CSV_PATH, crop: str = "cardamomo") -> pd.Series:
    """Annual export volume in kg, indexed by calendar year."""
    t = pd.read_csv(path, index_col="year")
    s = t[f"{crop}_mm_qq"] * 1e6 * QUINTAL_KG
    if not s.index.is_monotonic_increasing:
        raise ValueError("banguat: years out of order")
    if (s <= 0).any() or s.max() > 2e8:  # >200kt/yr would be a parse bug
        raise ValueError("banguat: implausible volumes — check extraction")
    return s.rename("gtm_exp_kg")


def annual_features(annual_kg: pd.Series) -> pd.DataFrame:
    """Per-year features, using only same-or-prior years per row.

    gtm_vol_yoy      log change vs prior year (supply momentum)
    gtm_vol_deficit  shortfall vs the trailing 5y mean of PRIOR years;
                     positive = shortage (spike-prone global regime)
    """
    yoy = np.log(annual_kg).diff()
    norm = annual_kg.rolling(5, min_periods=3).mean().shift(1)
    deficit = -(annual_kg / norm - 1.0)
    return pd.DataFrame({"gtm_vol_yoy": yoy, "gtm_vol_deficit": deficit})


def to_daily_features(idx: pd.DatetimeIndex, annual_kg: pd.Series) -> pd.DataFrame:
    """Step-function daily features under the 01-Apr-(Y+1) publication rule."""
    feats = annual_features(annual_kg)
    pub_dates = pd.DatetimeIndex(
        [pd.Timestamp(year=int(y) + 1, **PUBLICATION) for y in feats.index]
    )
    out = pd.DataFrame(np.nan, index=idx, columns=GTM_FEATURE_COLS)
    for col in GTM_FEATURE_COLS:
        published = pd.Series(feats[col].to_numpy(), index=pub_dates).sort_index()
        out[col] = published.reindex(
            published.index.union(idx)
        ).ffill().reindex(idx).to_numpy()
    return out


def build_gtm_features(market: pd.DataFrame) -> pd.DataFrame:
    """Feature block on the market calendar (ready for build_features alt=)."""
    return to_daily_features(market.index, load_annual_kg())
