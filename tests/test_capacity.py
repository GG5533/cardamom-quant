"""Tests for the capacity module (offline)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.capacity import (  # noqa: E402
    capacity_report, market_capacity, participation_table,
)

IDX = pd.bdate_range("2025-01-01", periods=200)
RNG = np.random.default_rng(11)

SPOT = pd.DataFrame(
    {
        "spot_avg": RNG.uniform(2500, 3000, 200),
        "qty_sold": RNG.uniform(60_000, 120_000, 200),  # kg
        "qty_arrived": RNG.uniform(70_000, 130_000, 200),
    },
    index=IDX,
)


def test_market_capacity_scales_with_adv():
    cap = market_capacity(SPOT)
    # ~90t x ~Rs2750 ~ Rs 24-25 crore/day; 5% => ~1.2 crore
    assert 15e7 < cap["auction_median_daily_value_inr"] < 40e7
    assert abs(cap["spot_capacity_inr"] - cap["auction_median_daily_value_inr"] * 0.05) < 1


def test_market_capacity_with_futures():
    fut = pd.DataFrame(
        {"fut_oi": 300.0, "fut_close": 2900.0}, index=IDX
    )
    cap = market_capacity(SPOT, fut)
    # 300 lots x 100kg x 2900 = Rs 8.7 crore OI; 10% => 0.87 crore
    assert abs(cap["fut_oi_value_inr"] - 300 * 100 * 2900) < 1
    assert abs(cap["fut_capacity_inr"] - cap["fut_oi_value_inr"] * 0.10) < 1


def test_participation_monotone_in_capital():
    pos = pd.Series(RNG.uniform(-1, 1, 200), index=IDX)
    tab = participation_table(pos, SPOT)
    med = tab["median_participation_pct"].to_numpy()
    assert (np.diff(med) > 0).all()          # more capital -> more participation
    assert tab["days_over_5pct_adv"].iloc[-1] >= tab["days_over_5pct_adv"].iloc[0]


def test_small_capital_fits_easily():
    pos = pd.Series(0.5, index=IDX)          # constant position, zero turnover after day 1
    tab = participation_table(pos, SPOT, capitals_inr=(5e6,))
    assert tab["median_participation_pct"].iloc[0] < 0.5  # buy-and-hold ~ no flow


def test_capacity_report_renders():
    pos = pd.Series(RNG.uniform(-1, 1, 200), index=IDX)
    txt = capacity_report(pos, SPOT)
    assert "auction median daily value" in txt
    assert "participation by capital" in txt
