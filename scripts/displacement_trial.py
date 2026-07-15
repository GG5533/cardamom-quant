"""Edge hunt, round 4 — feature displacement, ONE pre-registered trial.

    python scripts/displacement_trial.py

Three dilution results (T5, T7, and the original alt-block finding) said
the marginal feature must displace a weaker one, not join it
(RESULTS_REAL.md roadmap). T8 is the first displacement trial, declared
here before running:

  T8 physics-displaced-gbm
      Hypothesis: the champion configuration (core + auction-physics
      features, calibrated GBM, tranched 5d — "physics-gbm", Sharpe +0.79)
      carries redundant momentum horizons; on ~500 effective independent
      bets, dropping the two weakest (mom_10, mom_63 — both negative
      permutation dAUC on the last fold, and both correlated with
      mom_5/mom_21) should reduce variance without losing signal.
      T8 = champion feature set MINUS mom_10 and mom_63. Exactly ONE
      configuration — no sweep over which features to drop.

Sequencing disclosed: the choice of mom_10/mom_63 was motivated by
last-fold permutation-dAUC diagnostics of the already-counted champion —
i.e. this trial was formed AFTER looking at diagnostics of a prior trial,
like T5 in edge_hunt.py. It is counted like any other trial; that is what
the deflated Sharpe is for.

Leakage: NO new features are introduced — T8's feature matrix is a strict
subset of the champion's, every column of which already has a causality
test (tests/test_pipeline.py, tests/test_edge_hunt.py). No new leakage
test is required.

Champion parity: the champion is defined on the PRE-rain core, so the
round-3 rain columns (rain_mm, rain_climatology, rain_anomaly) are dropped
from the market frame before build_features — rain_anom_30/90 come out
all-NaN and are removed with the other dark columns, exactly as in the
round-2 run.

Ledger: 31 prior (24 + round-2's 5 + round-3's 2) + this 1 = 32.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd  # noqa: E402

from edge_hunt import PRIOR_LEDGER, calibrated_fold_proba, evaluate_stream  # noqa: E402
from run import load_dataset  # noqa: E402
from src.analysis.robustness import deflated_sharpe  # noqa: E402
from src.features.auction_physics import build_physics_features  # noqa: E402
from src.features.engineering import build_features  # noqa: E402
from src.validation.walkforward import PurgedWalkForward  # noqa: E402

logging.basicConfig(level=logging.WARNING)

ROUND2_SHARPES = [0.787, 0.672, -0.222, -0.174, 0.710]  # T1..T5, edge_hunt.py
ROUND3_SHARPES = [0.643, 0.546]                          # T6..T7, rain_trial.py

DROPPED = ["mom_10", "mom_63"]  # pre-registered; the ONLY configuration run


def main() -> None:
    market, tag = load_dataset(False)

    # Champion is defined on the PRE-rain core: remove the round-3 columns
    # so rain_anom_30/90 build as all-NaN and drop out, as in round 2.
    market = market.drop(columns=["rain_mm", "rain_climatology", "rain_anomaly"])

    X, y = build_features(market, alt=build_physics_features(market))
    X = X.drop(columns=X.columns[X.isna().all()])
    assert not {"rain_anom_30", "rain_anom_90"} & set(X.columns), \
        "rain leaked into the pre-rain champion core"
    champion_cols = list(X.columns)

    X = X.drop(columns=DROPPED)
    assert {"mom_5", "mom_21"} <= set(X.columns), "surviving momentum missing"
    print(f"champion matrix {len(champion_cols)} cols -> T8 {X.shape[1]} cols "
          f"(dropped {', '.join(DROPPED)})")

    daily = market["spot_avg"].pct_change().reindex(X.index)
    cv = PurgedWalkForward(n_splits=6, min_train=max(400, len(X) // 4))
    ledger = list(PRIOR_LEDGER) + ROUND2_SHARPES + ROUND3_SHARPES

    name = "T8 physics-displaced-gbm"
    results = {
        name: evaluate_stream(
            name, calibrated_fold_proba(X, y, cv), y, daily, ledger
        )
    }

    print(f"\n=== EDGE HUNT ROUND 4: DISPLACEMENT [{tag}] — tranched 5d vehicle ===")
    print("    incumbent champion: physics-gbm +0.79 (DSR 0.63 over 31 trials)\n")
    rows = [
        {
            "trial": k, "sharpe": m.get("sharpe"), "ci_5": m.get("ci_5"),
            "ci_95": m.get("ci_95"), "p_leq_0": m.get("p_leq_0"),
            "max_dd": m.get("max_dd"), "auc": m.get("auc"),
            "hit_vs_base": m.get("hit_vs_base_pts"),
        }
        for k, m in results.items()
    ]
    print(pd.DataFrame(rows).set_index("trial").round(3).to_string())

    dsr = deflated_sharpe(results[name]["net"].dropna(), ledger)
    print(f"\n  {name} — DSR over {dsr['n_trials']} trials: "
          f"{dsr['dsr']:.3f} (luck max {dsr['expected_max_sharpe']:.2f})")
    n_prior = len(PRIOR_LEDGER) + len(ROUND2_SHARPES) + len(ROUND3_SHARPES)
    print(f"  Ledger grew {n_prior} -> {len(ledger)}.")
    print("  Rule: T8 displaces the champion only if it beats +0.79 here; "
          "otherwise it is a counted kill.")


if __name__ == "__main__":
    main()
