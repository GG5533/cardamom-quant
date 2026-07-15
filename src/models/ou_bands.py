"""Optimal entry bands for trading an Ornstein-Uhlenbeck anomaly.

The Kalman layer hands us an anomaly following dX = -kappa X dt + sigma dW.
The strategy is a band policy: enter against the anomaly at |X| = b, exit
at 0. The band is not a knob to sweep — it is *derived* from the process:

    profit per cycle  =  b - c                (log-return captured, net cost)
    cycle time        =  E[T 0->±b] + E[T b->0]
    b* = argmax_b (b - c) / cycle_time(b)

Expected first-passage times come from the OU generator's classical
double-integral solution (Darling & Siegert): for a < b, hitting b from a,

    E[T] = (2/sigma^2) * ∫_a^b exp(V(y)) [∫_{-inf}^y exp(-V(z)) dz] dy,
    V(x) = kappa x^2 / sigma^2

evaluated numerically on a grid. Deriving the band from (kappa, sigma, cost)
instead of sweeping it keeps the DSR trial count at ONE for this policy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def expected_passage_time(
    a: float, b: float, kappa: float, sigma: float, n_grid: int = 400
) -> float:
    """E[time for OU to first hit b starting from a], a < b, in process time
    units (auction days here)."""
    if not (b > a):
        raise ValueError("requires a < b")
    sigma_st = sigma / np.sqrt(2 * kappa)
    lo = min(a, -8 * sigma_st)
    z = np.linspace(lo, b, n_grid)
    v = kappa * z**2 / sigma**2
    inner = np.concatenate([[0.0], np.cumsum(
        0.5 * (np.exp(-v[1:]) + np.exp(-v[:-1])) * np.diff(z)
    )])
    y_mask = z >= a
    integrand = np.exp(np.clip(v[y_mask], None, 700.0)) * inner[y_mask]
    return float(2 / sigma**2 * np.trapezoid(integrand, z[y_mask]))


def optimal_band(
    kappa: float, sigma_st: float, roundtrip_cost: float,
    grid: np.ndarray | None = None,
) -> tuple[float | None, float]:
    """Band maximizing net profit rate; (None, 0) if no band is profitable.

    kappa: per-day reversion rate; sigma_st: stationary anomaly std;
    roundtrip_cost: entry+exit cost in log-return units.
    """
    if kappa <= 0 or sigma_st <= 0:
        return None, 0.0
    sigma = sigma_st * np.sqrt(2 * kappa)
    if grid is None:
        grid = sigma_st * np.linspace(0.25, 3.0, 24)
    best_b, best_rate = None, 0.0
    for b in grid:
        if b <= roundtrip_cost:
            continue
        cycle = (
            expected_passage_time(0.0, b, kappa, sigma)      # 0 -> ±b
            + expected_passage_time(-b, 0.0, kappa, sigma)   # b -> 0 (symmetric)
        )
        if cycle <= 0:
            continue
        rate = (b - roundtrip_cost) / cycle
        if rate > best_rate:
            best_b, best_rate = float(b), float(rate)
    return best_b, best_rate


def band_policy_weights(anomaly: pd.Series, band: float | None) -> pd.Series:
    """Enter -sign(anomaly) at |anomaly| >= band, exit at the zero-crossing.

    Consumers execute weights next day (the backtest lags positions), so
    using day-t anomaly here is causal.
    """
    w = pd.Series(0.0, index=anomaly.index)
    if band is None:
        return w
    pos = 0.0
    for i, a in enumerate(anomaly.to_numpy()):
        if not np.isfinite(a):
            w.iloc[i] = pos
            continue
        if pos == 0.0:
            if a >= band:
                pos = -1.0
            elif a <= -band:
                pos = 1.0
        elif (pos < 0 and a <= 0) or (pos > 0 and a >= 0):
            pos = 0.0
            # allow immediate re-entry on an overshoot through the far band
            if a >= band:
                pos = -1.0
            elif a <= -band:
                pos = 1.0
        w.iloc[i] = pos
    return w
