"""Capacity analysis — could this strategy actually be run, and at what size?

A Sharpe means nothing without a capacity number attached. Cardamom is a
niche market: the e-auctions move ~40–130 tonnes/day (₹15–35 crore at
current prices) and the relaunched future carries a few hundred lots of
open interest. This module turns backtest positions into participation
rates against those real limits.

Conventions:
  * spot capacity ~ participation in daily auction traded VALUE
    (qty_sold x avg price);
  * futures capacity ~ share of open interest (lots of 100 kg);
  * rule-of-thumb ceilings (documented, adjustable): 5% of ADV / 10% of OI
    before impact costs invalidate the 15bps assumption.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LOT_KG = 100.0
MAX_ADV_PART = 0.05   # 5% of auction daily value
MAX_OI_PART = 0.10    # 10% of futures open interest


def market_capacity(
    spot_daily: pd.DataFrame,
    futures: pd.DataFrame | None = None,
    lookback: int = 63,
) -> dict:
    """Recent tradable-size envelope from real market data."""
    recent = spot_daily.tail(lookback)
    adv_value = (recent["qty_sold"] * recent["spot_avg"]).median()
    out = {
        "auction_median_daily_value_inr": float(adv_value),
        "auction_median_daily_kg": float(recent["qty_sold"].median()),
        "spot_capacity_inr": float(adv_value * MAX_ADV_PART),
    }
    if futures is not None and len(futures):
        f = futures.tail(lookback)
        oi_lots = f["fut_oi"].median()
        oi_value = oi_lots * LOT_KG * f["fut_close"].median()
        out.update(
            {
                "fut_median_oi_lots": float(oi_lots),
                "fut_oi_value_inr": float(oi_value),
                "fut_capacity_inr": float(oi_value * MAX_OI_PART),
            }
        )
    return out


def participation_table(
    positions: pd.Series,
    spot_daily: pd.DataFrame,
    capitals_inr: tuple[float, ...] = (5e6, 2.5e7, 1e8, 5e8),
) -> pd.DataFrame:
    """For each assumed capital: turnover-driven daily participation in the
    auction value, and the fraction of days above the 5% ADV ceiling.

    positions: backtest weight series (fraction of capital, signed).
    """
    daily_value = (spot_daily["qty_sold"] * spot_daily["spot_avg"]).reindex(
        positions.index
    )
    trade_frac = positions.diff().abs().fillna(positions.abs())
    rows = []
    for cap in capitals_inr:
        traded_inr = trade_frac * cap
        part = (traded_inr / daily_value).replace([np.inf, -np.inf], np.nan).dropna()
        rows.append(
            {
                "capital_inr_crore": cap / 1e7,
                "median_participation_pct": float(part.median() * 100),
                "p95_participation_pct": float(part.quantile(0.95) * 100),
                "days_over_5pct_adv": float((part > MAX_ADV_PART).mean() * 100),
            }
        )
    return pd.DataFrame(rows).set_index("capital_inr_crore").round(2)


def capacity_report(
    positions: pd.Series,
    spot_daily: pd.DataFrame,
    futures: pd.DataFrame | None = None,
) -> str:
    cap = market_capacity(spot_daily, futures)
    tab = participation_table(positions, spot_daily)
    lines = [
        f"auction median daily value : Rs {cap['auction_median_daily_value_inr']/1e7:,.1f} crore "
        f"({cap['auction_median_daily_kg']/1000:,.1f} t)",
        f"spot capacity @5% ADV      : Rs {cap['spot_capacity_inr']/1e7:,.1f} crore",
    ]
    if "fut_capacity_inr" in cap:
        lines.append(
            f"futures capacity @10% OI   : Rs {cap['fut_capacity_inr']/1e7:,.1f} crore "
            f"({cap['fut_median_oi_lots']:,.0f} lots OI)"
        )
    lines.append("\nparticipation by capital (crore INR):")
    lines.append(tab.to_string())
    return "\n".join(lines)
