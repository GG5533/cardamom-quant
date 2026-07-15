"""Leakage + sanity tests for the edge-hunt round: auction physics,
Kalman seasonal decomposition, OU band policy (offline)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest.engine import BacktestConfig, run_weights_backtest  # noqa: E402
from src.features.auction_physics import (  # noqa: E402
    PHYSICS_FEATURE_COLS, build_physics_features, hurst_rs, inventory_overhang,
)
from src.models.kalman_seasonal import fit_params, filter_anomaly, kalman_features  # noqa: E402
from src.models.ou_bands import band_policy_weights, optimal_band  # noqa: E402


def _market(n=900, seed=5):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2021-01-01", periods=n)
    px = pd.Series(2500 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=idx)
    arrived = pd.Series(rng.uniform(40_000, 120_000, n), index=idx)
    return pd.DataFrame(
        {
            "spot_avg": px,
            "spot_max": px * rng.uniform(1.1, 1.4, n),
            "qty_arrived": arrived,
            "qty_sold": arrived * rng.uniform(0.85, 1.0, n),
            "n_sessions": 2,
        }
    )


# ------------------------------------------------------------------ physics
def test_physics_features_are_strictly_causal():
    """Mutating TODAY's auction must not change TODAY's features."""
    m = _market()
    base = build_physics_features(m)
    mutated = m.copy()
    mutated.iloc[-1, :] = [9999.0, 19999.0, 1.0, 1.0, 1]
    after = build_physics_features(mutated)
    last = m.index[-1]
    for col in PHYSICS_FEATURE_COLS:
        b, a = base.loc[last, col], after.loc[last, col]
        assert (np.isnan(b) and np.isnan(a)) or b == a, col


def test_inventory_overhang_norm_is_past_years_only():
    m = _market(n=1000)
    base = inventory_overhang(m["qty_arrived"])
    # first observed crop year has no history -> NaN by construction
    cy_first_end = m.index[m.index < "2021-08-01"]
    assert base.loc[cy_first_end].isna().all()
    # doubling arrivals in the LAST crop year must not move earlier years
    mutated = m["qty_arrived"].copy()
    late = m.index >= "2023-08-01"
    mutated.loc[late] *= 2
    after = inventory_overhang(mutated)
    early = m.index < "2023-08-01"
    pd.testing.assert_series_equal(base[early], after[early])


def test_hurst_separates_trending_from_reverting():
    rng = np.random.default_rng(7)
    n = 500
    trending = np.cumsum(rng.normal(0.3, 1.0, n))          # persistent drift
    ar = np.zeros(n)
    for t in range(1, n):                                   # strong reversion
        ar[t] = -0.6 * ar[t - 1] + rng.normal()
    assert hurst_rs(np.diff(trending)) > hurst_rs(np.diff(ar))


# ------------------------------------------------------------------- kalman
def _planted(n=1200, phi=0.9, seed=11):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    doy = idx.dayofyear.to_numpy()
    seasonal = 0.25 * np.sin(2 * np.pi * doy / 365.25)
    anom = np.zeros(n)
    for t in range(1, n):
        anom[t] = phi * anom[t - 1] + rng.normal(0, 0.02)
    log_px = 7.8 + seasonal + anom + rng.normal(0, 0.004, n)
    return pd.Series(np.exp(log_px), index=idx), pd.Series(anom, index=idx)


def test_kalman_filter_is_causal():
    px, _ = _planted()
    feats, _ = kalman_features(px, train_end=600)
    mutated = px.copy()
    mutated.iloc[-1] *= 1.5
    feats2, _ = kalman_features(mutated, train_end=600)
    last = px.index[-1]
    for col in feats.columns:
        assert feats.loc[last, col] == feats2.loc[last, col], col


def test_kalman_recovers_planted_anomaly():
    px, true_anom = _planted()
    p = fit_params(np.log(px), train_end=600)
    assert 0.8 < p.phi < 0.98                    # found the persistence
    kf = filter_anomaly(np.log(px), p)
    oos = slice(600, None)
    corr = np.corrcoef(kf["anom"].iloc[oos], true_anom.iloc[oos])[0, 1]
    assert corr > 0.6


# ----------------------------------------------------------------- OU bands
def test_optimal_band_widens_with_cost():
    b_cheap, r_cheap = optimal_band(kappa=0.08, sigma_st=0.05, roundtrip_cost=0.002)
    b_dear, r_dear = optimal_band(kappa=0.08, sigma_st=0.05, roundtrip_cost=0.02)
    assert b_cheap is not None and r_cheap > 0
    assert b_dear is None or b_dear > b_cheap
    assert r_dear <= r_cheap


def test_band_policy_enters_and_exits():
    idx = pd.bdate_range("2024-01-01", periods=8)
    anom = pd.Series([0.0, 0.06, 0.04, 0.01, -0.001, 0.0, -0.07, -0.02], index=idx)
    w = band_policy_weights(anom, band=0.05)
    assert list(w) == [0, -1, -1, -1, 0, 0, 1, 1]


def test_run_weights_backtest_mechanics():
    idx = pd.bdate_range("2024-01-01", periods=5)
    r = pd.Series([0.0, 0.01, 0.01, 0.01, 0.01], index=idx)
    w = pd.Series([0.0, 1.0, 1.0, 0.0, 0.0], index=idx)
    bt = run_weights_backtest(w, r, BacktestConfig(cost_bps=15))
    assert list(bt["position"]) == [0, 0, 1, 1, 0]       # executed next day
    assert np.isclose(bt["cost"].sum(), 2 * 0.0015)      # enter + exit
    assert np.isclose(bt["net_ret"].iloc[2], 0.01 - 0.0015)
