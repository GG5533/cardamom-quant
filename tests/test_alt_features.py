"""Tests for calendar and climate-index feature modules (all offline)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.calendars import add_calendar_features  # noqa: E402
from src.data.climate_indices import (  # noqa: E402
    parse_oni,
    parse_psl_data,
    to_daily_features,
)

# real rows from the verified CPC file (fetched 2026-07-02)
ONI_SAMPLE = """\
 SEAS  YR   TOTAL   ANOM
  OND 2025  26.05  -0.55
  NDJ 2025  25.97  -0.54
  DJF 2026  26.13  -0.37
  JFM 2026  26.57  -0.16
"""

PSL_SAMPLE = """\
 2024 2025
 2024   0.10   0.20   0.30   0.40   0.50   0.60   0.70   0.80   0.90   1.00   1.10   1.20
 2025   -0.10  -0.20  -0.30  -99.90  -0.50  -0.60  -0.70  -0.80  -0.90  -1.00  -1.10  -1.20
 -99.90
 DMI HadISST1.1 provenance line
"""


# ------------------------------------------------------------------ parsing
def test_parse_oni_real_rows():
    s = parse_oni(ONI_SAMPLE)
    assert len(s) == 4
    assert s[pd.Timestamp(2026, 1, 1)] == -0.37   # DJF centred on Jan
    assert s[pd.Timestamp(2025, 12, 1)] == -0.54  # NDJ centred on Dec


def test_parse_psl_data_with_sentinel():
    s = parse_psl_data(PSL_SAMPLE, name="dmi")
    assert s[pd.Timestamp(2024, 6, 1)] == 0.60
    assert pd.Timestamp(2025, 4, 1) not in s.index  # -99.90 dropped
    assert len(s) == 23


# ------------------------------------------------- publication-lag leakage
def test_oni_publication_lag_no_lookahead():
    oni = parse_oni(ONI_SAMPLE)
    idx = pd.date_range("2026-01-01", "2026-05-31", freq="D")
    feats = to_daily_features(idx, oni)
    # DJF-2026 (centred Jan) publishes with 2m lag -> usable from Mar 1
    assert np.isnan(feats.loc["2026-02-15", "oni"]) or feats.loc[
        "2026-02-15", "oni"
    ] != -0.37
    assert feats.loc["2026-03-15", "oni"] == -0.37
    # JFM (centred Feb) -> usable from Apr
    assert feats.loc["2026-04-10", "oni"] == -0.16


def test_oni_regime_flags():
    oni = pd.Series(
        {pd.Timestamp(2023, 9, 1): 1.6, pd.Timestamp(2023, 10, 1): 1.8},
        name="oni",
    )
    idx = pd.date_range("2023-11-01", "2023-12-31", freq="D")
    feats = to_daily_features(idx, oni)
    assert (feats["oni_elnino"].dropna() == 1.0).all()
    assert (feats["oni_lanina"].dropna() == 0.0).all()


# --------------------------------------------------------------- calendars
def test_ramadan_countdown_and_window():
    # Ramadan 2026 starts Feb 18 (tabulated)
    idx = pd.date_range("2025-12-01", "2026-03-31", freq="D")
    f = add_calendar_features(idx)
    assert f.loc["2026-02-18", "days_to_ramadan"] == 0
    assert f.loc["2026-02-04", "days_to_ramadan"] == 14
    # stocking window: 14–56 days ahead -> Jan 2026 is inside
    assert f.loc["2026-01-15", "ramadan_stocking"] == 1.0
    assert f.loc["2026-02-17", "ramadan_stocking"] == 0.0  # 1 day out: too close
    # proximity ramps toward 1 at the start date
    assert f.loc["2026-02-18", "ramadan_proximity"] == 1.0
    assert f.loc["2025-12-01", "ramadan_proximity"] < 0.3


def test_ramadan_drift_captured():
    """The whole point: the window sits in different Gregorian months in
    different years, which day-of-year features cannot represent."""
    f24 = add_calendar_features(pd.date_range("2024-01-01", "2024-12-31", freq="D"))
    f19 = add_calendar_features(pd.date_range("2019-01-01", "2019-12-31", freq="D"))
    peak24 = f24[f24["ramadan_stocking"] == 1.0].index.month.unique()
    peak19 = f19[f19["ramadan_stocking"] == 1.0].index.month.unique()
    assert set(peak24) != set(peak19)  # windows fall in different months


def test_diwali_window():
    idx = pd.date_range("2024-09-01", "2024-11-30", freq="D")
    f = add_calendar_features(idx)
    assert f.loc["2024-11-01", "days_to_diwali"] == 0  # Diwali 2024
    assert f.loc["2024-10-10", "diwali_window"] == 1.0  # 22 days ahead
    assert f.loc["2024-11-20", "diwali_window"] == 0.0  # after


def test_calendar_features_never_nan_inside_tabulation():
    idx = pd.date_range("2012-01-01", "2025-12-31", freq="D")
    f = add_calendar_features(idx)
    assert f["days_to_ramadan"].notna().all()
    assert f["days_to_diwali"].notna().all()
