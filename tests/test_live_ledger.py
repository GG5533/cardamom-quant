"""Tests for the tamper-evident forecast ledger + live feature builder
(offline)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.forecast import features_all_rows  # noqa: E402
from src.features.engineering import HORIZON, build_features  # noqa: E402
from src.features.auction_physics import build_physics_features  # noqa: E402
from src.live.ledger import ChainedCsv  # noqa: E402


def _ledger(tmp_path):
    return ChainedCsv(tmp_path / "chain.csv", ["auction_date", "p_up"])


def test_chain_appends_and_verifies(tmp_path):
    led = _ledger(tmp_path)
    led.append({"auction_date": "2026-07-16", "p_up": "0.6100"})
    led.append({"auction_date": "2026-07-17", "p_up": "0.5800"})
    assert led.verify_chain() == 2


def test_chain_detects_tampering(tmp_path):
    led = _ledger(tmp_path)
    led.append({"auction_date": "2026-07-16", "p_up": "0.6100"})
    led.append({"auction_date": "2026-07-17", "p_up": "0.5800"})
    df = pd.read_csv(led.path, dtype=str)
    df.loc[0, "p_up"] = "0.9900"  # rewrite history
    df.to_csv(led.path, index=False)
    with pytest.raises(ValueError, match="hash chain broken"):
        led.verify_chain()


def test_chain_detects_deletion(tmp_path):
    led = _ledger(tmp_path)
    for i in range(3):
        led.append({"auction_date": f"2026-07-{16 + i}", "p_up": "0.5000"})
    df = pd.read_csv(led.path, dtype=str)
    df.iloc[[0, 2]].to_csv(led.path, index=False)  # drop the middle row
    with pytest.raises(ValueError, match="hash chain broken"):
        led.verify_chain()


def test_missing_field_rejected(tmp_path):
    with pytest.raises(ValueError, match="missing"):
        _ledger(tmp_path).append({"auction_date": "2026-07-16"})


def _market(n=700, seed=9):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-01-01", periods=n)
    px = pd.Series(2500 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=idx)
    arrived = pd.Series(rng.uniform(40_000, 120_000, n), index=idx)
    return pd.DataFrame({
        "spot_avg": px, "spot_max": px * rng.uniform(1.1, 1.4, n),
        "qty_arrived": arrived, "qty_sold": arrived * 0.95, "n_sessions": 2,
    })


def test_tail_features_match_unextended():
    """The synthetic extension must not contaminate real rows' features."""
    m = _market()
    X_all, labeled = features_all_rows(m)
    X_ref, _ = build_features(m, alt=build_physics_features(m))
    X_ref = X_ref.drop(columns=X_ref.columns[X_ref.isna().all()])
    common = X_ref.index
    pd.testing.assert_frame_equal(X_all.loc[common], X_ref[X_all.columns])
    # and the tail rows exist exactly where labels do not
    assert (~labeled).sum() == HORIZON
    assert X_all.index[-1] == m.index[-1]
