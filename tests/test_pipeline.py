"""Leakage, CV and backtest tests for the modeling pipeline (offline)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest.engine import BacktestConfig, run_backtest  # noqa: E402
from src.data.synthetic import generate_market  # noqa: E402
from src.features.engineering import HORIZON, build_features  # noqa: E402
from src.validation.walkforward import PurgedWalkForward  # noqa: E402


def test_synthetic_schema_matches_real():
    m = generate_market(start="2020-01-01", end="2021-12-31")
    for col in ("spot_avg", "qty_arrived", "qty_sold", "basis_pct", "rain_anomaly"):
        assert col in m.columns
    assert m["spot_avg"].gt(0).all()
    assert (m["qty_sold"] <= m["qty_arrived"] + 1e-9).all()


def test_features_are_strictly_lagged():
    """Mutating day-t market data must not change day-t features."""
    m = generate_market(start="2019-01-01", end="2021-12-31")
    X, _ = build_features(m)
    t = X.index[-HORIZON - 1]  # a date with defined label
    m2 = m.copy()
    m2.loc[t, ["spot_avg", "basis_pct", "rain_anomaly"]] = [9999.0, 0.5, 50.0]
    X2, _ = build_features(m2)
    row_a = X.loc[t].drop(["doy_sin", "doy_cos", "lean_season"])
    row_b = X2.loc[t].drop(["doy_sin", "doy_cos", "lean_season"])
    pd.testing.assert_series_equal(row_a, row_b, check_names=False)


def test_label_is_forward():
    m = generate_market(start="2019-01-01", end="2020-12-31")
    X, y = build_features(m)
    t = X.index[100]
    px = m["spot_avg"]
    t_pos = m.index.get_loc(t)
    realized = px.iloc[t_pos + HORIZON] / px.iloc[t_pos] - 1
    assert y.loc[t] == float(realized > 0)


def test_walkforward_purge_and_order():
    cv = PurgedWalkForward(n_splits=4, min_train=300, purge=5, embargo=5)
    n = 1000
    prev_end = 0
    for tr, te in cv.split(n):
        assert tr.max() < te.min() - 9          # purge+embargo gap respected
        assert te.min() >= prev_end             # test blocks march forward
        assert len(np.intersect1d(tr, te)) == 0
        prev_end = te.max()
    assert prev_end <= n - 1


def test_backtest_costs_and_leverage():
    idx = pd.bdate_range("2022-01-01", periods=300)
    rng = np.random.default_rng(1)
    rets = pd.Series(rng.normal(0, 0.01, 300), index=idx)
    proba = pd.Series(rng.uniform(0.3, 0.7, 300), index=idx)
    bt = run_backtest(proba, rets, BacktestConfig(max_leverage=1.5, cost_bps=15))
    assert bt["position"].abs().max() <= 1.5 + 1e-9
    assert (bt["cost"] >= 0).all()
    # random signal after costs shouldn't mint money
    assert bt["net_ret"].mean() < 0.001


def test_backtest_perfect_foresight_wins():
    """Sanity: a clairvoyant signal must produce a positive Sharpe."""
    idx = pd.bdate_range("2022-01-01", periods=400)
    rng = np.random.default_rng(2)
    rets = pd.Series(rng.normal(0.0002, 0.012, 400), index=idx)
    # clairvoyant: knows tomorrow's sign (position is shifted 1 inside engine)
    proba = (rets.shift(-1) > 0).astype(float).clip(0.05, 0.95)
    bt = run_backtest(proba, rets, BacktestConfig())
    net = bt["net_ret"].dropna()
    assert net.mean() / net.std() * np.sqrt(252) > 2.0
