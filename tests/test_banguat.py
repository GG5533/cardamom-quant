"""Tests for the Banguat annual Guatemala-supply feature block (offline)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.banguat import (  # noqa: E402
    GTM_FEATURE_COLS, annual_features, load_annual_kg, to_daily_features,
)


def _annual(vals, start=2015):
    return pd.Series(
        vals, index=range(start, start + len(vals)), name="gtm_exp_kg", dtype=float
    )


def test_deficit_norm_uses_prior_years_only():
    """Mutating year Y must not change year Y-1's features (past-only norm)."""
    kg = _annual([30e6, 31e6, 29e6, 30e6, 32e6, 30e6])
    base = annual_features(kg)
    mutated = kg.copy()
    mutated.loc[2020] = 1e6  # crash the last year
    after = annual_features(mutated)
    pd.testing.assert_frame_equal(base.loc[:2019], after.loc[:2019])


def test_collapse_reads_as_positive_deficit():
    kg = _annual([30e6, 31e6, 29e6, 30e6, 32e6, 17e6])  # 2024-25-style crash
    f = annual_features(kg)
    assert f["gtm_vol_deficit"].iloc[-1] > 0.35
    assert f["gtm_vol_yoy"].iloc[-1] < -0.5  # log change


def test_publication_lag_is_enforced():
    """Year Y is invisible before 01-Apr-(Y+1)."""
    kg = _annual([30e6, 31e6, 29e6, 30e6, 32e6, 17e6])  # last year = 2020
    idx = pd.date_range("2020-11-01", "2021-06-30", freq="D")
    daily = to_daily_features(idx, kg)
    # before publication: still the 2019 print
    before = daily.loc["2021-03-31", "gtm_vol_yoy"]
    assert np.isclose(before, np.log(32e6 / 30e6))
    # from publication day: the 2020 crash appears
    after = daily.loc["2021-04-01", "gtm_vol_yoy"]
    assert np.isclose(after, np.log(17e6 / 32e6))


def test_daily_features_are_strictly_causal():
    """Mutating the newest year never changes rows before its publication."""
    kg = _annual([30e6, 31e6, 29e6, 30e6, 32e6, 17e6])
    idx = pd.date_range("2019-01-01", "2021-12-31", freq="D")
    base = to_daily_features(idx, kg)
    mutated = kg.copy()
    mutated.loc[2020] = 40e6
    after = to_daily_features(idx, mutated)
    cut = "2021-03-31"
    pd.testing.assert_frame_equal(base.loc[:cut], after.loc[:cut])
    assert (base.loc["2021-04-01":, "gtm_vol_yoy"]
            != after.loc["2021-04-01":, "gtm_vol_yoy"]).all()


def test_real_extraction_parses_and_shows_the_collapse():
    kg = load_annual_kg()
    assert kg.index[0] == 1994 and kg.index[-1] >= 2025
    assert 10e6 < kg.loc[2019] < 60e6          # plausible scale (kg)
    assert kg.loc[2025] / kg.loc[2023] < 0.45  # the 2024-25 collapse
    f = to_daily_features(pd.date_range("2015-01-01", "2026-07-01"), kg)
    assert list(f.columns) == GTM_FEATURE_COLS
    assert f.loc["2026-06-01", "gtm_vol_deficit"] > 0.4  # shortage regime now
