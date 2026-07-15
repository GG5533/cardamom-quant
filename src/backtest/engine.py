"""Backtest engine: conviction sizing, vol targeting, leverage cap, costs.

Mechanics (next-day execution):
  raw signal      s_t = 2*p_up - 1                      in [-1, 1]
  vol targeting   w_t = s_t * (target_vol / realized_vol_t), capped
  rebalancing     the target weight is adopted only on scheduled days
                  (every `rebalance_every` days) AND only when conviction
                  clears the gate (|p_t - 0.5| > conviction_threshold);
                  otherwise the previous weight is held. Defaults
                  (1, 0.0) reproduce plain daily rebalancing.
  execution       position held over t+1 uses info through t (signal is
                  already built from lagged features; we still lag the
                  position one more day for execution realism)
  costs           cost_bps per unit turnover |w_t - w_{t-1}|

Non-overlapping evaluation: signals are generated daily but the metrics
report period returns net of costs — turnover, not trade count, drives cost.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BacktestConfig:
    target_vol_annual: float = 0.15
    max_leverage: float = 2.0
    cost_bps: float = 15.0
    vol_lookback: int = 21
    rebalance_every: int = 1           # adopt new target every N days only
    conviction_threshold: float = 0.0  # trade only if |p - 0.5| > threshold


def run_backtest(
    proba_up: pd.Series,
    daily_returns: pd.Series,
    cfg: BacktestConfig = BacktestConfig(),
) -> pd.DataFrame:
    """proba_up: model P(up) indexed by date; daily_returns: same calendar."""
    idx = proba_up.index.intersection(daily_returns.index)
    p = proba_up.reindex(idx)
    r = daily_returns.reindex(idx)

    signal = (2.0 * p - 1.0).clip(-1, 1)

    realized = r.rolling(cfg.vol_lookback, min_periods=15).std() * np.sqrt(252)
    scale = (cfg.target_vol_annual / realized).clip(upper=cfg.max_leverage)
    target = (signal * scale).clip(-cfg.max_leverage, cfg.max_leverage)

    scheduled = np.zeros(len(idx), dtype=bool)
    scheduled[:: max(cfg.rebalance_every, 1)] = True
    trade = pd.Series(scheduled, index=idx) & target.notna()
    if cfg.conviction_threshold > 0:
        trade &= (p - 0.5).abs() > cfg.conviction_threshold
    weight = target.where(trade).ffill().fillna(0.0)

    pos = weight.shift(1).fillna(0.0)          # execute next day
    gross = pos * r
    turnover = pos.diff().abs().fillna(pos.abs())
    costs = turnover * cfg.cost_bps / 10_000.0
    net = gross - costs

    out = pd.DataFrame(
        {
            "proba_up": p,
            "signal": signal,
            "position": pos,
            "turnover": turnover,
            "gross_ret": gross,
            "cost": costs,
            "net_ret": net,
            "equity": (1 + net.fillna(0)).cumprod(),
        }
    )
    out.index.name = "date"
    return out
