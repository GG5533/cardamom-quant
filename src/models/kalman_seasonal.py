"""Structural decomposition of log spot: level + seasonal + AR(1) anomaly.

Ported/adapted from the Bloomberg-Jarvis world-model Kalman engine. The
model answers one question the momentum block can't: *how far is today's
price from its own slowly-drifting seasonal fair value, and how fast does
that gap decay?*

    log P_t = mu(doy_t; beta) + level_t + anom_t + obs noise
    level_t = level_{t-1} + eta          (random walk — regime shifts, 2019)
    anom_t  = phi * anom_{t-1} + eps     (AR(1) — the tradable reversion)

Estimation contract (leakage-safe): beta (2 Fourier harmonics), phi and all
variances are fitted on the TRAINING window only; the Kalman FILTER (never
the smoother — that peeks forward) is then run causally over the full
series. anom_t at time t uses observations up to t only; consumers shift(1)
like every other feature.

kappa = -ln(phi) per auction day is the mean-reversion rate the OU band
policy consumes; half-life = ln 2 / kappa.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

KALMAN_FEATURE_COLS = ["kf_anom_z", "kf_anom_chg"]


@dataclass
class KalmanParams:
    beta: np.ndarray          # [const, sin1, cos1, sin2, cos2]
    phi: float                # AR(1) coefficient of the anomaly
    q_level: float            # level state noise variance
    q_anom: float             # anomaly state noise variance
    r_obs: float              # observation noise variance
    sigma_st: float           # stationary anomaly std, sqrt(q_anom/(1-phi^2))

    @property
    def kappa(self) -> float:
        return float(-np.log(max(min(self.phi, 0.9999), 1e-6)))

    @property
    def half_life_days(self) -> float:
        return float(np.log(2) / self.kappa)


def _fourier(idx: pd.DatetimeIndex) -> np.ndarray:
    doy = idx.dayofyear.to_numpy()
    w = 2 * np.pi * doy / 365.25
    return np.column_stack(
        [np.ones_like(w), np.sin(w), np.cos(w), np.sin(2 * w), np.cos(2 * w)]
    )


def fit_params(log_px: pd.Series, train_end: int) -> KalmanParams:
    """Fit seasonal beta + AR(1)/variance parameters on [:train_end] only."""
    tr = log_px.iloc[:train_end].dropna()
    F = _fourier(tr.index)
    beta, *_ = np.linalg.lstsq(F, tr.to_numpy(), rcond=None)
    resid = tr.to_numpy() - F @ beta

    # AR(1) on the seasonal residual (Yule-Walker, lag 1)
    d = resid - resid.mean()
    phi = float(np.clip(np.dot(d[1:], d[:-1]) / np.dot(d[:-1], d[:-1]), 0.5, 0.995))
    sigma_st2 = float(np.var(d, ddof=1))
    q_anom = sigma_st2 * (1 - phi**2)

    # observation noise ~ high-frequency auction chatter; level walk slow
    dd = np.diff(resid)
    r_obs = 0.25 * float(np.var(dd, ddof=1)) / 2.0
    q_level = 1e-4 * sigma_st2

    return KalmanParams(
        beta=beta, phi=phi, q_level=q_level, q_anom=q_anom,
        r_obs=max(r_obs, 1e-8), sigma_st=np.sqrt(sigma_st2),
    )


def filter_anomaly(log_px: pd.Series, p: KalmanParams) -> pd.DataFrame:
    """Causal Kalman filter over the full series with train-fitted params.

    Returns frame with anom (filtered mean), anom_sd (filtered std), level.
    """
    y = (log_px.to_numpy() - _fourier(log_px.index) @ p.beta)
    T = len(y)
    A = np.array([[1.0, 0.0], [0.0, p.phi]])          # [level, anom]
    C = np.array([[1.0, 1.0]])
    Q = np.diag([p.q_level, p.q_anom])
    R = np.array([[p.r_obs]])

    x = np.zeros(2)
    P = np.diag([p.sigma_st**2, p.sigma_st**2])
    anom = np.empty(T)
    anom_sd = np.empty(T)
    level = np.empty(T)
    for t in range(T):
        # predict
        x = A @ x
        P = A @ P @ A.T + Q
        if np.isfinite(y[t]):
            # update
            S = C @ P @ C.T + R
            K = P @ C.T @ np.linalg.inv(S)
            x = x + (K @ (y[t] - C @ x)).ravel()
            P = (np.eye(2) - K @ C) @ P
        level[t], anom[t] = x
        anom_sd[t] = np.sqrt(max(P[1, 1], 1e-16))

    return pd.DataFrame(
        {"anom": anom, "anom_sd": anom_sd, "level": level}, index=log_px.index
    )


def kalman_features(px: pd.Series, train_end: int) -> tuple[pd.DataFrame, KalmanParams]:
    """Per-fold feature builder: fit on [:train_end], filter causally, lag 1.

    kf_anom_z   — anomaly in filtered-uncertainty units (the reversion signal)
    kf_anom_chg — 5d change of the anomaly (is the gap opening or closing?)
    """
    log_px = np.log(px)
    p = fit_params(log_px, train_end)
    kf = filter_anomaly(log_px, p)
    out = pd.DataFrame(index=px.index)
    out["kf_anom_z"] = (kf["anom"] / kf["anom_sd"].clip(lower=1e-8)).shift(1)
    out["kf_anom_chg"] = kf["anom"].diff(5).shift(1)
    return out[KALMAN_FEATURE_COLS], p
