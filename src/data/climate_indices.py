"""Climate teleconnection features — LEADING indicators of the monsoon.

The rainfall loader measures weather that already happened. ENSO and the
Indian Ocean Dipole lead the Indian summer monsoon by months: El Niño years
tend to suppress it, positive IOD tends to enhance it (NOAA climate.gov,
IMD seasonal outlooks). For a crop priced off Kerala yield expectations,
that turns weather from a lagging into a *forecastable* driver — months of
extra look-ahead the synthetic model never had.

Sources (both free, plain-text, decades of history):
  * ONI  — NOAA CPC, https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt
           (verified 2026-07-02, current through JFM-2026; 3-month centred
           seasons DJF/JFM/... with TOTAL and ANOM columns)
  * DMI  — NOAA PSL, https://psl.noaa.gov/gcos_wgsp/Timeseries/Data/dmi.had.long.data
           (classic PSL layout: "startyr endyr" header, then "year v1..v12",
           sentinel value in the header's footer block)

LEAKAGE DISCIPLINE — the part worth showcasing: ONI's 3-month season centred
on month M is only published early in month M+1, and the centred window
itself contains month M+1 data. Features therefore use PUBLICATION-lagged
values: the value mapped to any trading day is the latest season that was
fully published at that time (2-month shift from the centre month). Same
treatment for DMI (1-month publication shift on monthly values).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ONI_URL = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
DMI_URL = "https://psl.noaa.gov/gcos_wgsp/Timeseries/Data/dmi.had.long.data"

# ONI season code -> centre month
_SEASON_CENTRE = {
    "DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4, "AMJ": 5, "MJJ": 6,
    "JJA": 7, "JAS": 8, "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12,
}


# ---------------------------------------------------------------------- ONI
def parse_oni(text: str) -> pd.Series:
    """CPC oni.ascii.txt -> monthly Series of anomalies (index = centre month)."""
    rows = []
    for line in text.strip().splitlines():
        parts = line.split()
        if len(parts) == 4 and parts[0] in _SEASON_CENTRE:
            seas, yr, _total, anom = parts
            rows.append(
                (pd.Timestamp(int(yr), _SEASON_CENTRE[seas], 1), float(anom))
            )
    if not rows:
        raise ValueError("ONI: no parseable rows — format changed?")
    s = pd.Series(dict(rows)).sort_index()
    s.name = "oni"
    return s


def fetch_oni(url: str = ONI_URL, timeout: int = 30) -> pd.Series:
    import requests

    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return parse_oni(r.text)


# ---------------------------------------------------------------------- DMI
def parse_psl_data(text: str, name: str = "dmi", sentinel_hint: float = -99.0) -> pd.Series:
    """Classic NOAA-PSL .data layout -> monthly Series.

    Layout: first line 'startyear endyear'; then one row per year with 12
    monthly values; trailing footer lines (sentinel value, provenance).
    Any value <= sentinel_hint is treated as missing.
    """
    lines = text.strip().splitlines()
    try:
        y0, y1 = (int(x) for x in lines[0].split()[:2])
    except (ValueError, IndexError) as e:
        raise ValueError(f"{name}: unexpected PSL header: {lines[0]!r}") from e
    rows = {}
    for line in lines[1:]:
        parts = line.split()
        if len(parts) == 13:
            try:
                yr = int(parts[0])
            except ValueError:
                continue
            if y0 <= yr <= y1:
                for m, v in enumerate(parts[1:], start=1):
                    val = float(v)
                    rows[pd.Timestamp(yr, m, 1)] = (
                        np.nan if val <= sentinel_hint else val
                    )
    if not rows:
        raise ValueError(f"{name}: no parseable rows — format changed?")
    s = pd.Series(rows).sort_index().dropna()
    s.name = name
    return s


def fetch_dmi(url: str = DMI_URL, timeout: int = 30) -> pd.Series:
    import requests

    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return parse_psl_data(r.text, name="dmi")


# ----------------------------------------------------------- daily features
def to_daily_features(
    index: pd.DatetimeIndex,
    oni: pd.Series,
    dmi: pd.Series | None = None,
    oni_pub_lag_months: int = 2,
    dmi_pub_lag_months: int = 1,
) -> pd.DataFrame:
    """Map monthly indices onto a trading calendar, publication-lagged.

    A value centred on month M becomes usable from the first day of month
    M + lag. This is deliberately conservative — being one month late costs
    little on a multi-month teleconnection but guarantees no look-ahead.
    """
    df = pd.DataFrame(index=index)
    df.index.name = "date"

    lagged_oni = oni.copy()
    lagged_oni.index = lagged_oni.index + pd.DateOffset(months=oni_pub_lag_months)
    df["oni"] = lagged_oni.reindex(
        pd.DatetimeIndex(index.to_period("M").to_timestamp())
    ).to_numpy()
    df["oni_elnino"] = (df["oni"] >= 0.5).astype(float)
    df["oni_lanina"] = (df["oni"] <= -0.5).astype(float)

    if dmi is not None:
        lagged_dmi = dmi.copy()
        lagged_dmi.index = lagged_dmi.index + pd.DateOffset(months=dmi_pub_lag_months)
        df["dmi"] = lagged_dmi.reindex(
            pd.DatetimeIndex(index.to_period("M").to_timestamp())
        ).to_numpy()
    else:
        df["dmi"] = np.nan
    return df
