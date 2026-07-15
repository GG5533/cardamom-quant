"""Tests for horizon-matched rebalancing, conviction gating and isotonic
calibration (offline)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.calibration import brier_score, isotonic_calibrator  # noqa: E402
from src.backtest.engine import BacktestConfig, run_backtest  # noqa: E402


def _inputs(n=300, seed=3):
    idx = pd.bdate_range("2022-01-01", periods=n)
    rng = np.random.default_rng(seed)
    rets = pd.Series(rng.normal(0, 0.01, n), index=idx)
    proba = pd.Series(rng.uniform(0.2, 0.8, n), index=idx)
    return proba, rets


# ----------------------------------------------------------------- rebalance
def test_default_config_matches_daily_rebalance_spec():
    """Defaults must reproduce the published daily-rebalanced engine."""
    proba, rets = _inputs()
    cfg = BacktestConfig()
    bt = run_backtest(proba, rets, cfg)
    signal = (2 * proba - 1).clip(-1, 1)
    realized = rets.rolling(cfg.vol_lookback, min_periods=15).std() * np.sqrt(252)
    scale = (cfg.target_vol_annual / realized).clip(upper=cfg.max_leverage)
    expected = (signal * scale).clip(-cfg.max_leverage, cfg.max_leverage)
    expected = expected.fillna(0.0).shift(1).fillna(0.0)
    pd.testing.assert_series_equal(bt["position"], expected, check_names=False)


def test_rebalance_cadence_holds_position_between_trades():
    proba, rets = _inputs()
    bt = run_backtest(proba, rets, BacktestConfig(rebalance_every=5))
    # weight adopted at indices 0,5,10,...; executed (shift 1) at 1,6,11,...
    moves = np.flatnonzero(bt["position"].diff().abs().to_numpy() > 1e-12)
    assert len(moves) > 10  # it does trade
    assert all((i - 1) % 5 == 0 for i in moves)


def test_slower_cadence_cuts_turnover():
    proba, rets = _inputs()
    fast = run_backtest(proba, rets, BacktestConfig(rebalance_every=1))
    slow = run_backtest(proba, rets, BacktestConfig(rebalance_every=5))
    assert slow["turnover"].sum() < fast["turnover"].sum() * 0.5


# ---------------------------------------------------------------- conviction
def test_conviction_gate_blocks_low_conviction():
    proba, rets = _inputs()
    lukewarm = proba.clip(0.42, 0.58)  # never clears a 0.10 gate
    bt = run_backtest(lukewarm, rets, BacktestConfig(conviction_threshold=0.10))
    assert (bt["position"] == 0).all()
    assert bt["cost"].sum() == 0.0


def test_conviction_gate_trades_only_on_conviction():
    proba, rets = _inputs()
    bt = run_backtest(proba, rets, BacktestConfig(conviction_threshold=0.15))
    gated = run_backtest(proba, rets, BacktestConfig())
    assert 0 < bt["turnover"].sum() < gated["turnover"].sum()
    # every position change traces to a day the gate was open
    moves = np.flatnonzero(bt["position"].diff().abs().to_numpy() > 1e-12)
    open_gate = (proba - 0.5).abs().to_numpy() > 0.15
    assert all(open_gate[i - 1] for i in moves)


def test_execution_stays_next_day_causal():
    """Mutating TODAY's proba must not change TODAY's position."""
    proba, rets = _inputs()
    cfg = BacktestConfig(rebalance_every=5, conviction_threshold=0.05)
    base = run_backtest(proba, rets, cfg)
    mutated = proba.copy()
    mutated.iloc[-1] = 0.99
    after = run_backtest(mutated, rets, cfg)
    assert base["position"].iloc[-1] == after["position"].iloc[-1]


# --------------------------------------------------------------- calibration
def test_isotonic_calibrator_fixes_overconfidence():
    rng = np.random.default_rng(11)
    n = 2000
    # true edge is mild (55/45) but the raw model screams 0.05/0.95
    y = pd.Series((rng.uniform(size=n) < 0.5).astype(float))
    p_raw = pd.Series(np.where(y.to_numpy() > 0.5, 0.55, 0.45))
    p_raw += rng.normal(0, 0.02, n)
    overconfident = pd.Series(np.clip(0.5 + 4.0 * (p_raw - 0.5), 0.01, 0.99))
    iso = isotonic_calibrator(overconfident[: n // 2], y[: n // 2])
    fixed = pd.Series(iso.predict(overconfident[n // 2 :]), index=y.index[n // 2 :])
    assert brier_score(y[n // 2 :], fixed) < brier_score(y[n // 2 :], overconfident[n // 2 :])
    assert fixed.between(0, 1).all()
