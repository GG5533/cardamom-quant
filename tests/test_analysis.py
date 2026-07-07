"""Tests for robustness, calibration and interpretability modules."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.calibration import (  # noqa: E402
    brier_score, calibration_summary, calibration_table, enso_phase,
    regime_performance,
)
from src.analysis.robustness import (  # noqa: E402
    block_bootstrap_sharpe, deflated_sharpe, expected_max_sharpe_annual,
    probabilistic_sharpe,
)

IDX = pd.bdate_range("2020-01-01", periods=1000)
RNG = np.random.default_rng(3)


def _rets(mean):
    return pd.Series(RNG.normal(mean, 0.01, len(IDX)), index=IDX)


# ------------------------------------------------------------------ bootstrap
def test_bootstrap_ci_brackets_point_estimate():
    r = _rets(0.0008)  # strong strategy
    b = block_bootstrap_sharpe(r, n_boot=500)
    assert b["ci_5"] < b["sharpe"] < b["ci_95"]
    assert b["p_leq_0"] < 0.05


def test_bootstrap_flags_noise_as_noise():
    r = _rets(0.0)
    b = block_bootstrap_sharpe(r, n_boot=500)
    assert b["p_leq_0"] > 0.10  # cannot reject luck


# ----------------------------------------------------------------- psr / dsr
def test_psr_high_for_signal_low_for_noise():
    assert probabilistic_sharpe(_rets(0.001)) > 0.95
    assert 0.2 < probabilistic_sharpe(_rets(0.0)) < 0.8


def test_expected_max_sharpe_grows_with_trials():
    trials = [0.5, -0.2, 0.8, 0.1, -0.4, 0.3]
    few = expected_max_sharpe_annual(2, trials[:2])
    many = expected_max_sharpe_annual(6, trials)
    assert many > few >= 0


def test_dsr_below_psr():
    """The multiple-testing haircut must only ever hurt."""
    r = _rets(0.0006)
    d = deflated_sharpe(r, [1.2, 0.3, -0.5, 0.9, 0.1, 0.7])
    assert d["dsr"] <= d["psr_vs_zero"]
    assert d["n_trials"] == 6


# --------------------------------------------------------------- calibration
def test_perfectly_calibrated_probs_have_small_gap():
    p = pd.Series(RNG.uniform(0.1, 0.9, 5000))
    y = pd.Series((RNG.uniform(0, 1, 5000) < p).astype(float))
    tab = calibration_table(y, p)
    assert tab["gap"].abs().max() < 0.06
    s = calibration_summary(y, p)
    assert s["brier"] < s["brier_climatology"]


def test_brier_score_bounds():
    y = pd.Series([1.0, 0.0, 1.0])
    assert brier_score(y, pd.Series([1.0, 0.0, 1.0])) == 0.0
    assert brier_score(y, pd.Series([0.0, 1.0, 0.0])) == 1.0


# -------------------------------------------------------------------- regimes
def test_enso_phase_thresholds():
    oni = pd.Series([0.6, -0.6, 0.1], index=pd.RangeIndex(3))
    assert list(enso_phase(oni)) == ["elnino", "lanina", "neutral"]


def test_regime_performance_splits():
    idx = pd.bdate_range("2021-01-01", periods=400)
    net = pd.Series(RNG.normal(0.0003, 0.01, 400), index=idx)
    y = pd.Series(RNG.integers(0, 2, 400).astype(float), index=idx)
    p = pd.Series(RNG.uniform(0.3, 0.7, 400), index=idx)
    regs = pd.DataFrame(
        {"season": np.where(idx.month.isin([3, 4, 5, 6, 7]), "lean", "harvest")},
        index=idx,
    )
    tab = regime_performance(net, y, p, regs)
    assert set(tab.index) == {"season=lean", "season=harvest"}
    assert (tab["n_days"] >= 30).all()


# ----------------------------------------------------------------------- shap
def test_shap_on_gbm_if_available():
    shap = pytest.importorskip("shap")  # noqa: F841
    from src.analysis.interpret import mean_abs_shap, shap_values_for
    from src.models.baselines import make_gbm

    n = 500
    X = pd.DataFrame(
        {"a": RNG.normal(size=n), "b": RNG.normal(size=n), "noise": RNG.normal(size=n)}
    )
    y = (X["a"] + 0.5 * X["b"] + RNG.normal(0, 0.5, n) > 0).astype(float)
    gbm = make_gbm().fit(X.to_numpy(), y)
    sv = shap_values_for("gbm", gbm, X)
    ma = mean_abs_shap(sv)
    assert ma.index[0] == "a"           # strongest driver ranks first
    assert ma["a"] > ma["noise"] * 2    # and clearly beats noise
