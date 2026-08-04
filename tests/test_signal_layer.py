"""Tests for the cross-market/macro/microstructure signal layer (offline)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.comtrade import GuatemalaExportsLoader, parse_comtrade  # noqa: E402
from src.data.fx import parse_fred_csv, to_daily_features as fx_daily  # noqa: E402
from src.data.loaders import build_market_dataset  # noqa: E402
from src.features.alt_features import (  # noqa: E402
    ALT_FEATURE_COLS,
    auction_microstructure,
    build_alt_features,
)
from src.features.engineering import build_features  # noqa: E402

COMTRADE_FIXTURE = json.dumps(
    {
        "data": [
            {"period": 202410, "cmdCode": "090831", "netWgt": 3_000_000, "primaryValue": 90_000_000},
            {"period": 202410, "cmdCode": "090832", "netWgt": 200_000, "primaryValue": 5_000_000},
            {"period": 202411, "cmdCode": "090831", "netWgt": 2_500_000, "primaryValue": 80_000_000},
        ]
    }
)

FRED_FIXTURE = """DATE,DEXINUS
2026-06-25,86.10
2026-06-26,86.25
2026-06-29,.
2026-06-30,86.40
"""


# ------------------------------------------------------------------ comtrade
def test_parse_comtrade_sums_hs_codes():
    m = parse_comtrade(COMTRADE_FIXTURE)
    assert m.loc["2024-10-01", "exp_kg"] == 3_200_000
    assert m.loc["2024-11-01", "exp_usd"] == 80_000_000
    assert len(m) == 2


def test_supply_shock_positive_on_collapse():
    # 4 stable years then a 45% crop failure (the 2024-25 scenario)
    months = pd.date_range("2020-01-01", "2024-12-01", freq="MS")
    kg = pd.Series(1_000_000.0, index=months)
    kg.iloc[-6:] = 550_000.0
    monthly = pd.DataFrame({"exp_kg": kg, "exp_usd": kg * 30})
    feats = GuatemalaExportsLoader.add_features(monthly)
    assert feats["supply_shock"].iloc[-1] > 1.0  # strong shortage signal
    assert abs(feats["exp_kg_yoy"].iloc[-1] - (-0.45)) < 1e-9


def test_comtrade_publication_lag():
    months = pd.date_range("2024-01-01", "2024-12-01", freq="MS")
    monthly = pd.DataFrame(
        {
            "exp_kg": 1.0,
            "exp_usd": 1.0,
            "exp_kg_yoy": np.linspace(0, 1, len(months)),
            "supply_shock": 0.5,
        },
        index=months,
    )
    idx = pd.date_range("2024-01-01", "2025-03-31", freq="D")
    daily = GuatemalaExportsLoader.to_daily_features(idx, monthly)
    # Jan-2024 print (yoy=0.0) usable from Apr-2024, not before
    assert np.isnan(daily.loc["2024-03-15", "gtm_exp_kg_yoy"])
    assert daily.loc["2024-04-15", "gtm_exp_kg_yoy"] == 0.0


# ----------------------------------------------------------------------- fx
def test_parse_fred_csv_missing_dot():
    s = parse_fred_csv(FRED_FIXTURE)
    assert len(s) == 3  # '.' dropped
    assert s[pd.Timestamp("2026-06-30")] == 86.40


def test_fx_features_ffill_limited():
    s = parse_fred_csv(FRED_FIXTURE)
    idx = pd.date_range("2026-06-25", "2026-06-30", freq="D")
    f = fx_daily(idx, s)
    assert f.loc["2026-06-28", "usdinr"] == 86.25  # weekend ffill
    assert f.loc["2026-06-30", "usdinr"] == 86.40


# ------------------------------------------------------------ microstructure
def _spot(n=400, seed=7):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-01", periods=n)
    arrived = pd.Series(rng.uniform(40_000, 120_000, n), index=idx)
    sold = arrived * rng.uniform(0.85, 1.0, n)
    return pd.DataFrame(
        {
            "spot_avg": rng.uniform(2000, 3000, n),
            "spot_max": 3500.0,
            "qty_arrived": arrived,
            "qty_sold": sold,
            "n_sessions": 2,
        },
        index=idx,
    )


def test_microstructure_is_strictly_causal():
    """Mutating TODAY's auction must not change TODAY's features."""
    spot = _spot()
    base = auction_microstructure(spot)
    mutated = spot.copy()
    mutated.iloc[-1, mutated.columns.get_loc("qty_sold")] = 1.0  # crash today
    after = auction_microstructure(mutated)
    last = spot.index[-1]
    for col in ("tension", "tension_z_63", "arrivals_surprise"):
        b, a = base.loc[last, col], after.loc[last, col]
        assert (np.isnan(b) and np.isnan(a)) or b == a


def test_tension_bounded_and_sane():
    f = auction_microstructure(_spot())
    t = f["tension"].dropna()
    assert t.between(0, 1.5).all()
    assert 0.8 < t.mean() < 1.0  # ~90% sell-through in the fixture


# --------------------------------------------------------------------- build
def test_build_alt_features_schema_stable_without_optional_feeds():
    f = build_alt_features(_spot())
    assert list(f.columns) == ALT_FEATURE_COLS
    assert f["oni"].isna().all()          # not wired -> explicit NaN
    assert f["gtm_supply_shock"].isna().all()
    assert f["ramadan_stocking"].notna().all()  # calendars always present


def test_build_alt_features_with_climate():
    oni = pd.Series(
        {pd.Timestamp(2024, m, 1): 0.1 * m for m in range(1, 13)}, name="oni"
    )
    f = build_alt_features(_spot(), oni=oni)
    assert f["oni"].notna().sum() > 0
    assert list(f.columns) == ALT_FEATURE_COLS


# ------------------------------------------------------------------ MCX basis
def _basis_market():
    """Real-shaped spot+futures frame (as build_market_dataset would join
    them, post the MCX relaunch backfill), long enough for basis_chg's
    diff(5) to be defined mid-series.
    """
    idx = pd.bdate_range("2026-01-01", periods=40)
    rng = np.random.default_rng(0)
    spot_px = 3000.0 + np.cumsum(rng.normal(0, 10, len(idx)))
    spot = pd.DataFrame(
        {
            "spot_avg": spot_px,
            "spot_max": spot_px * 1.1,
            "qty_arrived": 100_000.0,
            "qty_sold": 95_000.0,
            "n_sessions": 2,
        },
        index=idx,
    )
    fut_px = spot_px * 1.02 + rng.normal(0, 5, len(idx))
    futures = pd.DataFrame(
        {
            "fut_close": fut_px,
            "fut_volume": 100.0,
            "fut_oi": 300.0,
            "contract": "2026-12-31",
            "days_to_expiry": np.arange(len(idx), 0, -1),
            "fut_ret": pd.Series(fut_px, index=idx).pct_change(),
            "fut_cont": fut_px,
            "regime": "post2025",
        },
        index=idx,
    )
    rain = pd.DataFrame(
        {"rain_mm": np.nan, "rain_climatology": np.nan, "rain_anomaly": np.nan},
        index=idx,
    )
    return build_market_dataset(spot=spot, futures=futures, rain=rain)


def test_basis_features_are_strictly_causal():
    """Mutating TODAY's basis (as build_market_dataset would compute it from
    a changed futures close / spot print) must not change TODAY's
    basis_pct / basis_chg features (build_features shifts both by 1 day),
    but MUST change the NEXT day's -- otherwise the shift is a no-op.
    """
    m = _basis_market()
    X, _ = build_features(m)
    assert X["basis_pct"].notna().any(), "fixture produced no basis coverage"

    t = X.index[-6]  # mid-series, defined label, defined basis_chg
    t_next = X.index[X.index.get_loc(t) + 1]

    mutated = m.copy()
    # simulate a changed futures print at t propagating through the same
    # honest join build_market_dataset performs: basis/basis_pct move too.
    mutated.loc[t, "fut_close"] = 99999.0
    mutated.loc[t, "basis"] = mutated.loc[t, "fut_close"] - mutated.loc[t, "spot_avg"]
    mutated.loc[t, "basis_pct"] = mutated.loc[t, "basis"] / mutated.loc[t, "spot_avg"]
    X2, _ = build_features(mutated)

    for col in ("basis_pct", "basis_chg"):
        b, a = X.loc[t, col], X2.loc[t, col]
        assert (np.isnan(b) and np.isnan(a)) or b == a, f"{col} leaked at t"
        # the mutation must actually propagate forward (shift(1) is live)
        nb, na = X.loc[t_next, col], X2.loc[t_next, col]
        assert not ((np.isnan(nb) and np.isnan(na)) or nb == na), (
            f"{col} at t+1 did not react to a t mutation -- shift(1) is dead"
        )
