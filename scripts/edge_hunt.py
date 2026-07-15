"""Edge hunt, round 2 — four pre-registered trials, counted before running.

    python scripts/edge_hunt.py

The tranched-5d calibrated book (RESULTS_REAL.md, 07-Jul update) sits at
Sharpe +0.57 / DSR 0.49 — a maybe. These four hypotheses each add an
ingredient the pipeline has never seen; each is ONE trial in the deflated-
Sharpe ledger, declared here before any result was known:

  T1 physics-gbm    core + auction-physics features (bid dispersion,
                    Kaldor-Working inventory overhang, rolling Hurst),
                    calibrated GBM in the tranched 5d vehicle
  T2 kalman-gbm     core + causal Kalman anomaly features (per-fold fitted
                    structural decomposition), same vehicle
  T3 ou-bands       standalone OU band policy on the Kalman anomaly; the
                    band is DERIVED per fold from (kappa, sigma, cost) via
                    first-passage calculus, not swept
  T4 conformal-gbm  GBM 5d-return regressor gated by a split-conformal
                    80% interval (trade only when it excludes zero), same
                    vehicle
  T5 physics+kalman declared AFTER T1/T2 both cleared zero on the first
                    run (T1 +0.79, T2 +0.67): the two blocks together.
                    Sequencing disclosed; the trial is counted like any
                    other — that is what the deflated Sharpe is for.

Ledger: 24 prior trials (6 scorecard + 16 grid + 2 tranched) + these 5 = 29.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from horizon_experiment import tranched_net_returns  # noqa: E402
from run import load_dataset  # noqa: E402
from src.analysis.robustness import block_bootstrap_sharpe, deflated_sharpe  # noqa: E402
from src.backtest.engine import BacktestConfig, run_weights_backtest  # noqa: E402
from src.features.auction_physics import build_physics_features  # noqa: E402
from src.features.engineering import HORIZON, build_features, forward_returns  # noqa: E402
from src.metrics import backtest_metrics, classification_metrics  # noqa: E402
from src.models.baselines import make_gbm  # noqa: E402
from src.models.kalman_seasonal import kalman_features  # noqa: E402
from src.models.ou_bands import band_policy_weights, optimal_band  # noqa: E402
from src.validation.walkforward import PurgedWalkForward  # noqa: E402

try:  # sklearn >= 1.0
    from sklearn.ensemble import HistGradientBoostingRegressor
except ImportError:  # pragma: no cover
    HistGradientBoostingRegressor = None

from sklearn.isotonic import IsotonicRegression  # noqa: E402

logging.basicConfig(level=logging.WARNING)

# Every configuration ever evaluated on this dataset (RESULTS_REAL.md):
# 6 scorecard trials + 16 horizon-grid cells + 2 tranched books.
PRIOR_LEDGER = [
    0.300, -0.205, 0.123, 0.300, -0.114, -0.223,            # scorecard
    0.123, 0.210, 0.459, 0.459, 0.774, 0.588, 0.532, 0.545,  # raw grid
    0.161, 0.040, 0.241, 0.120, 0.930, 0.308, 0.177, -0.024,  # calibrated grid
    0.480, 0.570,                                             # tranched books
]

CALIB_FRACTION = 0.2
ROUNDTRIP_COST = 2 * 15 / 10_000.0  # enter + exit one unit at 15bps


def calibrated_fold_proba(X, y, cv):
    """Per-fold isotonic-calibrated GBM stream for a fold-varying X builder.

    X may be a DataFrame (static features) or a callable train_end -> frame
    aligned to y.index (fold-fitted features, e.g. the Kalman block).
    """
    proba = pd.Series(np.nan, index=y.index)
    for tr, te in cv.split(len(y)):
        Xf = X(tr[-1] + 1) if callable(X) else X
        arr = Xf.to_numpy()
        n_cal = max(50, int(len(tr) * CALIB_FRACTION))
        inner = tr[: -(n_cal + HORIZON)]
        calib = tr[-n_cal:]
        base = make_gbm()
        base.fit(arr[inner], y.iloc[inner])
        iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
        iso.fit(base.predict_proba(arr[calib])[:, 1], y.iloc[calib].to_numpy())
        proba.iloc[te] = iso.predict(base.predict_proba(arr[te])[:, 1])
    return proba


def evaluate_stream(name, proba, y, daily, ledger):
    mask = proba.notna()
    net, _ = tranched_net_returns(proba[mask], daily[mask])
    m = backtest_metrics(net)
    if net.dropna().std() > 0:
        m.update(block_bootstrap_sharpe(net))
    m.update(classification_metrics(y[mask], proba[mask]))
    if m["sharpe"] == m["sharpe"]:
        ledger.append(m["sharpe"])
    m["net"] = net
    return m


def main() -> None:
    market, tag = load_dataset(False)
    X_core, y = build_features(market, alt=None)
    X_core = X_core.drop(columns=X_core.columns[X_core.isna().all()])
    daily = market["spot_avg"].pct_change().reindex(X_core.index)
    px = market["spot_avg"].reindex(X_core.index)
    fwd = forward_returns(market).reindex(X_core.index)
    cv = PurgedWalkForward(n_splits=6, min_train=max(400, len(X_core) // 4))

    ledger = list(PRIOR_LEDGER)
    results: dict[str, dict] = {}

    # ---- T1: auction physics ------------------------------------------------
    phys = build_physics_features(market)
    X1, y1 = build_features(market, alt=phys)
    X1 = X1.drop(columns=X1.columns[X1.isna().all()])
    results["T1 physics-gbm"] = evaluate_stream(
        "T1", calibrated_fold_proba(X1, y1, cv), y1, daily, ledger
    )

    # ---- T2: Kalman anomaly features (fold-fitted) --------------------------
    kf_params = {}

    def x_with_kalman(train_end: int) -> pd.DataFrame:
        feats, p = kalman_features(px, train_end)
        kf_params[train_end] = p
        return X_core.join(feats)

    results["T2 kalman-gbm"] = evaluate_stream(
        "T2", calibrated_fold_proba(x_with_kalman, y, cv), y, daily, ledger
    )

    # ---- T3: OU band policy on the Kalman anomaly ---------------------------
    from src.models.kalman_seasonal import filter_anomaly, fit_params

    weights = pd.Series(0.0, index=X_core.index)
    bands = []
    for tr, te in cv.split(len(X_core)):
        p = fit_params(np.log(px), tr[-1] + 1)
        kf = filter_anomaly(np.log(px), p)
        sigma_st = float(kf["anom"].iloc[: tr[-1] + 1].std(ddof=1))
        band, rate = optimal_band(p.kappa, sigma_st, ROUNDTRIP_COST)
        bands.append((band, p.half_life_days))
        w = band_policy_weights(kf["anom"], band)
        weights.iloc[te] = w.iloc[te]
    bt = run_weights_backtest(weights, daily, BacktestConfig())
    oos = weights.index[np.concatenate([te for _, te in cv.split(len(X_core))])]
    net3 = bt["net_ret"].reindex(oos)
    m3 = backtest_metrics(net3)
    m3.update(block_bootstrap_sharpe(net3) if net3.std() > 0 else {})
    m3["net"] = net3
    m3["bands"] = bands
    if m3["sharpe"] == m3["sharpe"]:  # never-trades fold set -> NaN stays out
        ledger.append(m3["sharpe"])
    results["T3 ou-bands"] = m3

    # ---- T4: conformal-gated GBM regressor ----------------------------------
    proba4 = pd.Series(np.nan, index=y.index)
    arr = X_core.to_numpy()
    fwd_np = fwd.to_numpy()
    for tr, te in cv.split(len(X_core)):
        n_cal = max(50, int(len(tr) * CALIB_FRACTION))
        inner = tr[: -(n_cal + HORIZON)]
        calib = tr[-n_cal:]
        reg = HistGradientBoostingRegressor(
            max_depth=3, learning_rate=0.05, max_iter=300,
            l2_regularization=1.0, random_state=0,
        )
        reg.fit(arr[inner], fwd_np[inner])
        q = float(np.quantile(np.abs(fwd_np[calib] - reg.predict(arr[calib])), 0.8))
        yhat = reg.predict(arr[te])
        gate = np.abs(yhat) > q  # 80% interval excludes zero
        proba4.iloc[te] = 0.5 + 0.4 * np.sign(yhat) * gate
    results["T4 conformal-gbm"] = evaluate_stream(
        "T4", proba4, y, daily, ledger
    )

    # ---- T5: physics + kalman combined (sequenced trial — see docstring) ----
    X1_aligned = X1.reindex(y.index)

    def x_combo(train_end: int) -> pd.DataFrame:
        feats, _ = kalman_features(px, train_end)
        return X1_aligned.join(feats)

    results["T5 physics+kalman"] = evaluate_stream(
        "T5", calibrated_fold_proba(x_combo, y, cv), y, daily, ledger
    )

    # ---- report -------------------------------------------------------------
    print(f"\n=== EDGE HUNT ROUND 2 [{tag}] — tranched 5d vehicle, after costs ===")
    print("    incumbent: tranched calibrated core/gbm Sharpe +0.57, DSR 0.49\n")
    rows = []
    for name, m in results.items():
        rows.append(
            {
                "trial": name, "sharpe": m.get("sharpe"),
                "ci_5": m.get("ci_5"), "ci_95": m.get("ci_95"),
                "p_leq_0": m.get("p_leq_0"), "max_dd": m.get("max_dd"),
                "auc": m.get("auc"), "hit_vs_base": m.get("hit_vs_base_pts"),
            }
        )
    print(pd.DataFrame(rows).set_index("trial").round(3).to_string())

    if results["T3 ou-bands"].get("bands"):
        b = results["T3 ou-bands"]["bands"]
        print("\n  T3 per-fold (band, anomaly half-life days):",
              [(None if x is None else round(x, 4), round(h, 1)) for x, h in b])

    best_name = max(results, key=lambda k: results[k].get("sharpe") or -9)
    best = results[best_name]
    dsr = deflated_sharpe(best["net"].dropna(), ledger)
    print(f"\n=== BEST NEW TRIAL: {best_name} ===")
    print(f"  Sharpe {best['sharpe']:+.2f}  90% CI [{best.get('ci_5', float('nan')):+.2f}, "
          f"{best.get('ci_95', float('nan')):+.2f}]  p(SR<=0) {best.get('p_leq_0', float('nan')):.3f}")
    print(f"  DSR over {dsr['n_trials']} trials: {dsr['dsr']:.3f} "
          f"(expected max SR from luck {dsr['expected_max_sharpe']:.2f})")
    print(f"\n  Ledger grew {len(PRIOR_LEDGER)} -> {len(ledger)}. "
          "Every cell above is now a counted trial forever.")


if __name__ == "__main__":
    main()
