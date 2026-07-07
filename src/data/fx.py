"""FX competitiveness — USD/INR from FRED (no API key needed).

Indian cardamom competes with Guatemalan supply for Gulf demand in USD terms;
a weaker rupee makes Indian offers more competitive. FRED's plain CSV
endpoint (verified reachable 2026-07-02) requires no key:

    https://fred.stlouisfed.org/graph/fredgraph.csv?id=DEXINUS

Format: header ``DATE,DEXINUS`` (newer exports may use ``observation_date``),
one row per business day, ``.`` for missing. DEXINUS is the H.10 noon
buying rate, INR per USD.

USD/GTQ has no FRED daily series; Banguat publishes it but scraping a
Spanish-language central-bank portal is poor ROI — the Guatemalan supply
effect is captured more directly by the Comtrade volume loader. Documented
as a deliberate scope cut.

Features are simple and lagged by construction (levels/returns of a daily
close known same-evening): fx level, 21d momentum, 63d z-score.
"""
from __future__ import annotations

import io
import logging

import pandas as pd

logger = logging.getLogger(__name__)

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv"
USDINR_SERIES = "DEXINUS"


def parse_fred_csv(text: str, series: str = USDINR_SERIES) -> pd.Series:
    """fredgraph.csv -> daily Series; '.' means missing."""
    df = pd.read_csv(io.StringIO(text))
    date_col = df.columns[0]  # 'DATE' or 'observation_date'
    if series not in df.columns:
        raise ValueError(f"fred: column {series!r} not in {list(df.columns)}")
    s = pd.Series(
        pd.to_numeric(df[series], errors="coerce").to_numpy(),
        index=pd.to_datetime(df[date_col]),
        name="usdinr",
    ).dropna()
    if s.empty:
        raise ValueError("fred: no numeric observations parsed")
    return s.sort_index()


def fetch_usdinr(timeout: int = 60) -> pd.Series:
    import requests

    r = requests.get(FRED_CSV, params={"id": USDINR_SERIES}, timeout=timeout)
    r.raise_for_status()
    return parse_fred_csv(r.text)


def to_daily_features(index: pd.DatetimeIndex, usdinr: pd.Series) -> pd.DataFrame:
    """FX features on the trading calendar (ffill weekends/holidays, max 5d)."""
    fx = usdinr.reindex(index.union(usdinr.index)).ffill(limit=5).reindex(index)
    df = pd.DataFrame(index=index)
    df.index.name = "date"
    df["usdinr"] = fx
    df["inr_mom_21"] = fx.pct_change(21)
    mu = fx.rolling(63, min_periods=40).mean()
    sd = fx.rolling(63, min_periods=40).std()
    df["inr_z_63"] = (fx - mu) / sd
    return df
