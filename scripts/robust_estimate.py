"""Slicing-robust champion estimate — kill the fold-boundary wobble.

    python scripts/robust_estimate.py

Extending the dataset by seven days moved the champion's Sharpe +0.79 →
+0.69 for one reason: walk-forward fold boundaries shift with n, every
GBM refits, and a single fold layout is one draw from a distribution of
equally-defensible layouts. This script replaces the single-layout point
estimate with a declared family of SIX layouts:

    n_splits ∈ {5, 6, 7}  ×  min_train ∈ {base, base + 63}

and reports (a) the per-layout Sharpes with their range, and (b) the
LAYOUT-BLENDED stream — per-day mean of the calibrated probabilities of
every layout that scores that day out-of-sample — as the headline, with
its own bootstrap CI.

This is estimator refinement, NOT trial selection: the family is fixed
here ex ante, nothing is picked by outcome, and the blend averages over
all six. The DSR trial ledger is unchanged; RESULTS_REAL.md reports the
blended number as the champion's standing estimate.
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

from edge_hunt import calibrated_fold_proba  # noqa: E402
from horizon_experiment import tranched_net_returns  # noqa: E402
from run import load_dataset  # noqa: E402
from src.analysis.robustness import block_bootstrap_sharpe  # noqa: E402
from src.features.auction_physics import build_physics_features  # noqa: E402
from src.features.engineering import build_features  # noqa: E402
from src.metrics import backtest_metrics, classification_metrics  # noqa: E402
from src.validation.walkforward import PurgedWalkForward  # noqa: E402

logging.basicConfig(level=logging.WARNING)

LAYOUTS = [(s, off) for s in (5, 6, 7) for off in (0, 63)]


def champion_estimate(market: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """The champion's slicing-robust estimate: per-layout table + blended metrics.

    Exposed as a function so nothing has to hardcode the champion's Sharpe to
    compare against. It moves with the dataset -- a literal copied from an
    earlier run silently becomes a wrong baseline (see ensemble_trial.py).
    """
    market = market.drop(columns=["rain_mm", "rain_climatology", "rain_anomaly"])
    X, y = build_features(market, alt=build_physics_features(market))
    X = X.drop(columns=X.columns[X.isna().all()])
    daily = market["spot_avg"].pct_change().reindex(X.index)
    base_min = max(400, len(X) // 4)

    streams, rows = [], []
    for n_splits, off in LAYOUTS:
        cv = PurgedWalkForward(n_splits=n_splits, min_train=base_min + off)
        proba = calibrated_fold_proba(X, y, cv)
        mask = proba.notna()
        net, _ = tranched_net_returns(proba[mask], daily[mask])
        m = backtest_metrics(net)
        rows.append({"layout": f"{n_splits} folds · min_train +{off}",
                     "oos_days": int(mask.sum()), "sharpe": m["sharpe"]})
        streams.append(proba)

    blend = pd.concat(streams, axis=1).mean(axis=1)
    mask = blend.notna()
    net, _ = tranched_net_returns(blend[mask], daily[mask])
    m = backtest_metrics(net)
    m.update(block_bootstrap_sharpe(net))
    m.update(classification_metrics(y[mask], blend[mask]))
    m["oos_days"] = int(mask.sum())
    return pd.DataFrame(rows).set_index("layout"), m


def main() -> None:
    market, tag = load_dataset(False)
    per, m = champion_estimate(market)

    print(f"\n=== SLICING-ROBUST CHAMPION ESTIMATE [{tag}] — 6 layouts ===")
    print(per.round(3).to_string())
    print(f"\n  layout mean {per['sharpe'].mean():+.3f}  "
          f"range [{per['sharpe'].min():+.2f}, {per['sharpe'].max():+.2f}]")

    print(f"\n  BLENDED stream ({m['oos_days']} OOS days):")
    print(f"    Sharpe {m['sharpe']:+.3f}  90% CI [{m['ci_5']:+.2f}, "
          f"{m['ci_95']:+.2f}]  p(SR<=0) {m['p_leq_0']:.3f}")
    print(f"    AUC {m['auc']:.3f}  hit vs base {m['hit_vs_base_pts']:+.1f}pts  "
          f"max_dd {m['max_dd']:+.1%}")
    print("\n  This blended figure is the champion's standing estimate; the "
          "single-layout numbers above show the wobble it removes.")


if __name__ == "__main__":
    main()
