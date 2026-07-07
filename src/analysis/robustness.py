"""Robustness statistics — is the Sharpe distinguishable from luck?

Three tools most portfolio projects never show:

  * circular block-bootstrap CI on the annualized Sharpe (blocks preserve
    the autocorrelation a daily-rebalanced 5d-horizon strategy has);
  * Probabilistic / Deflated Sharpe Ratio (Bailey & López de Prado):
    the probability the true Sharpe exceeds a benchmark once non-normality
    (skew, fat tails) and MULTIPLE TESTING (we tried several models) are
    accounted for;
  * an ablation harness: same pipeline, feature sets swapped, so
    "the alt data helps" is a measured claim, not a vibe.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

ANN = 252


# ------------------------------------------------------------- bootstrap CI
def block_bootstrap_sharpe(
    net_ret: pd.Series,
    n_boot: int = 2000,
    block: int = 21,
    seed: int = 0,
) -> dict:
    """Circular block bootstrap of the annualized Sharpe ratio."""
    r = net_ret.dropna().to_numpy()
    T = len(r)
    if T < 3 * block:
        raise ValueError(f"series too short ({T}) for block={block}")
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(T / block))
    sharpes = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, T, n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel() % T
        s = r[idx[:T]]
        sd = s.std()
        sharpes[i] = s.mean() / sd * np.sqrt(ANN) if sd > 0 else 0.0
    point = r.mean() / r.std() * np.sqrt(ANN)
    return {
        "sharpe": float(point),
        "ci_5": float(np.percentile(sharpes, 5)),
        "ci_95": float(np.percentile(sharpes, 95)),
        "p_leq_0": float((sharpes <= 0).mean()),
        "n_days": T,
    }


# ------------------------------------------------- probabilistic / deflated
def probabilistic_sharpe(
    net_ret: pd.Series, benchmark_sr_annual: float = 0.0
) -> float:
    """PSR: P(true SR > benchmark), adjusting for skew/kurtosis (B&LdP 2012).

    Computed in per-period units internally.
    """
    r = net_ret.dropna().to_numpy()
    T = len(r)
    sr = r.mean() / r.std()                      # per-period
    sr_b = benchmark_sr_annual / np.sqrt(ANN)    # per-period benchmark
    skew = stats.skew(r)
    kurt = stats.kurtosis(r, fisher=False)
    denom = np.sqrt(max(1 - skew * sr + (kurt - 1) / 4 * sr**2, 1e-12))
    z = (sr - sr_b) * np.sqrt(T - 1) / denom
    return float(stats.norm.cdf(z))


def expected_max_sharpe_annual(n_trials: int, trial_sharpes_annual: list[float]) -> float:
    """E[max SR] under n_trials of zero-true-skill strategies whose SR
    estimates have the empirical cross-trial variance (B&LdP 2014)."""
    if n_trials < 2:
        return 0.0
    v = np.var(np.asarray(trial_sharpes_annual) / np.sqrt(ANN), ddof=1)
    gamma = 0.5772156649
    e_max = np.sqrt(max(v, 1e-12)) * (
        (1 - gamma) * stats.norm.ppf(1 - 1 / n_trials)
        + gamma * stats.norm.ppf(1 - 1 / (n_trials * np.e))
    )
    return float(e_max * np.sqrt(ANN))


def deflated_sharpe(
    net_ret: pd.Series, all_trial_sharpes_annual: list[float]
) -> dict:
    """DSR: PSR against the expected-max-Sharpe of everything we tried.

    all_trial_sharpes_annual should include EVERY model/config evaluated —
    honesty about selection is the whole point.
    """
    n = len(all_trial_sharpes_annual)
    sr_star = expected_max_sharpe_annual(n, all_trial_sharpes_annual)
    return {
        "n_trials": n,
        "expected_max_sharpe": sr_star,
        "dsr": probabilistic_sharpe(net_ret, benchmark_sr_annual=sr_star),
        "psr_vs_zero": probabilistic_sharpe(net_ret, benchmark_sr_annual=0.0),
    }


# ------------------------------------------------------------------ ablation
def ablation_table(results: dict[str, dict]) -> pd.DataFrame:
    """results: {variant_name: metrics_dict} -> tidy comparison frame."""
    rows = []
    for name, m in results.items():
        rows.append(
            {
                "variant": name,
                "hit_vs_base_pts": m.get("hit_vs_base_pts"),
                "auc": m.get("auc"),
                "sharpe": m.get("sharpe"),
                "max_dd": m.get("max_dd"),
                "p_sharpe_leq_0": m.get("p_leq_0"),
            }
        )
    return pd.DataFrame(rows).set_index("variant")
