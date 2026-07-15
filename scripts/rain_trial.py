"""Edge hunt, round 3 — the rain feed, two pre-registered trials.

    python scripts/rain_trial.py     # requires merge_rain.py to have run

The weather thesis was the project's original alt-data centrepiece
(ALT_DATA_RESEARCH.md): monsoon anomalies over the Idukki belt lead yield,
yield leads price. The feed is now wired (IMD gridded, real publication
lag: recent months NaN). Two trials, declared before running:

  T6 rain-gbm          core features + rain_anom_30/90 (they were always
                       in CORE_FEATURE_COLS, dark until now), calibrated
                       GBM in the tranched 5d vehicle
  T7 physics+rain-gbm  the round-2 champion block + rain

Ledger: 29 prior (24 + round-2's 5) + these 2 = 31. The round-2 Sharpes
are pinned below from the committed edge_hunt.py output.
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


def main() -> None:
    market, tag = load_dataset(False)
    if market["rain_anomaly"].notna().sum() < 500:
        sys.exit("rain not wired — run scripts/merge_rain.py first")

    daily = None
    ledger = list(PRIOR_LEDGER) + ROUND2_SHARPES
    results = {}

    for name, alt in (
        ("T6 rain-gbm", None),
        ("T7 physics+rain-gbm", build_physics_features(market)),
    ):
        X, y = build_features(market, alt=alt)
        X = X.drop(columns=X.columns[X.isna().all()])
        if daily is None:
            daily = market["spot_avg"].pct_change().reindex(X.index)
        cv = PurgedWalkForward(n_splits=6, min_train=max(400, len(X) // 4))
        assert "rain_anom_30" in X.columns, "rain features still dark"
        results[name] = evaluate_stream(
            name, calibrated_fold_proba(X, y, cv), y, daily, ledger
        )

    print(f"\n=== EDGE HUNT ROUND 3: RAIN [{tag}] — tranched 5d vehicle ===")
    print("    champions: physics-gbm +0.79 (DSR 0.63), tranched core +0.57\n")
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

    best = max(results, key=lambda k: results[k].get("sharpe") or -9)
    dsr = deflated_sharpe(results[best]["net"].dropna(), ledger)
    print(f"\n  best: {best} — DSR over {dsr['n_trials']} trials: "
          f"{dsr['dsr']:.3f} (luck max {dsr['expected_max_sharpe']:.2f})")
    print(f"  Ledger grew {len(PRIOR_LEDGER) + len(ROUND2_SHARPES)} -> {len(ledger)}.")
    print("  Rule: rain stays only if it pays here; otherwise the feed is cut.")


if __name__ == "__main__":
    main()
