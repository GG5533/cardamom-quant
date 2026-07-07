"""Calibrated synthetic cardamom market — the methodology testbed.

NOT a market finding. This exists so the entire pipeline (features →
walk-forward → backtest) can be built, stress-tested and unit-tested before
real feeds are wired, and retained afterwards as an appendix. Every chart
from this generator is labeled [SYNTHETIC].

Structure mirrors the real dataset schema exactly (build_market_dataset
columns), with the drivers the real market exhibits:
  * multiplicative day-of-year seasonality (lean-season premium),
  * GARCH-like volatility clustering + Student-t shocks,
  * monsoon rainfall with wet/dry years; lagged adverse-weather drift
    (deficient rain -> higher prices with a lag) — the learnable signal,
  * cointegrated futures with mean-reverting basis,
  * auction quantities with harvest seasonality (arrivals peak Aug–Feb).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def generate_market(
    start: str = "2016-01-01",
    end: str = "2026-06-30",
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, end)
    n = len(idx)
    doy = idx.dayofyear.to_numpy()

    # --- rainfall: monsoon climatology + persistent wet/dry regimes --------
    clim = 4.0 + 14.0 * np.exp(-0.5 * ((doy - 200) / 55.0) ** 2)  # peaks ~Jul
    wet_state = np.zeros(n)
    state = 0.0
    for i in range(n):
        state = 0.995 * state + rng.normal(0, 0.3)
        wet_state[i] = state
    rain = np.maximum(clim * np.exp(0.35 * wet_state) + rng.gamma(1.2, 1.5, n) - 1.8, 0.0)
    rain_anom = rain - clim

    # --- price: seasonality + weather-driven drift + t-shocks --------------
    lean = np.isin(idx.month, [3, 4, 5, 6, 7]).astype(float)
    # lagged 60d mean rain anomaly, deficient rain -> upward drift
    ra = pd.Series(rain_anom, index=idx)
    ra_60 = ra.rolling(60, min_periods=30).mean().shift(20).fillna(0.0).to_numpy()
    drift = 0.00035 * lean - 0.00006 * ra_60 + 0.00028 * np.sign(-ra_60)

    # GARCH-ish vol
    sigma = np.empty(n)
    sigma[0] = 0.012
    eps = np.empty(n)
    for t in range(n):
        if t:
            sigma[t] = np.sqrt(
                0.05 * 0.012**2 + 0.90 * sigma[t - 1] ** 2 + 0.05 * eps[t - 1] ** 2
            )
        eps[t] = sigma[t] * rng.standard_t(df=4) / np.sqrt(4 / 2)
    seasonal_level = 1.0 + 0.10 * lean
    logp = np.log(1500.0) + np.cumsum(drift + eps)
    spot = np.exp(logp) * seasonal_level

    # --- auction quantities: harvest arrivals + demand-tension noise -------
    harvest = np.isin(idx.month, [8, 9, 10, 11, 12, 1, 2]).astype(float)
    arrivals = rng.lognormal(mean=10.6 + 0.5 * harvest, sigma=0.35)
    sell_through = np.clip(rng.normal(0.93 + 0.02 * lean, 0.05), 0.6, 1.0)

    # --- futures: cointegrated, mean-reverting basis ------------------------
    basis = np.empty(n)
    basis[0] = 0.03
    for t in range(1, n):
        basis[t] = 0.97 * basis[t - 1] + rng.normal(0, 0.006)
    fut = spot * (1.0 + 0.02 + basis)

    df = pd.DataFrame(
        {
            "spot_avg": spot,
            "spot_max": spot * rng.uniform(1.15, 1.35, n),
            "qty_arrived": arrivals,
            "qty_sold": arrivals * sell_through,
            "n_sessions": 2,
            "spot_staleness_days": 0,
            "fut_close": fut,
            "fut_volume": rng.integers(50, 400, n).astype(float),
            "fut_oi": rng.integers(100, 900, n).astype(float),
            "contract": "SYNTH",
            "days_to_expiry": 30,
            "fut_ret": pd.Series(fut, index=idx).pct_change().to_numpy(),
            "fut_cont": fut,
            "regime": "synthetic",
            "basis": fut - spot,
            "basis_pct": (fut - spot) / spot,
            "rain_mm": rain,
            "rain_climatology": clim,
            "rain_anomaly": rain_anom,
        },
        index=idx,
    )
    df.index.name = "date"
    return df
