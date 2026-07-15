"""Edge hunt, round 5 — Guatemala supply state, two pre-registered trials.

    python scripts/gtm_trial.py

The world's #1 exporter lost ~45% of its crop across 2024-25; Indian
prices should carry a global-supply premium when Guatemala is short. UN
Comtrade paywalled its mirror, so the feed comes from the origin — Banco
de Guatemala's annual export-volume series (src/data/banguat.py), step-
published every 01-Apr with the prior year's figure. Annual is coarse;
the mechanism (crop-year supply level) is slow, so it is honest coarse.

Declared before running:

  T9  gtm-gbm          core + gtm_vol_yoy + gtm_vol_deficit
  T10 physics+gtm-gbm  the champion block + the Guatemala state

Ledger: 32 prior (24 + round-2's 5 + rain's 2 + T8) + these 2 = 34.
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
from rain_trial import ROUND2_SHARPES  # noqa: E402
from run import load_dataset  # noqa: E402
from src.analysis.robustness import deflated_sharpe  # noqa: E402
from src.data.banguat import build_gtm_features  # noqa: E402
from src.features.auction_physics import build_physics_features  # noqa: E402
from src.features.engineering import build_features  # noqa: E402
from src.validation.walkforward import PurgedWalkForward  # noqa: E402

logging.basicConfig(level=logging.WARNING)

RAIN_SHARPES = [0.643, 0.546]   # T6, T7 (scripts/rain_trial.py output)
T8_SHARPE = [0.626]             # displacement kill (scripts/displacement_trial.py)


def main() -> None:
    market, tag = load_dataset(False)
    # champion parity: the physics arm stays on the pre-rain core definition
    market = market.drop(columns=["rain_mm", "rain_climatology", "rain_anomaly"])

    gtm = build_gtm_features(market)
    ledger = list(PRIOR_LEDGER) + ROUND2_SHARPES + RAIN_SHARPES + T8_SHARPE
    results = {}
    daily = None

    for name, alt in (
        ("T9 gtm-gbm", gtm),
        ("T10 physics+gtm-gbm", build_physics_features(market).join(gtm)),
    ):
        X, y = build_features(market, alt=alt)
        X = X.drop(columns=X.columns[X.isna().all()])
        assert "gtm_vol_deficit" in X.columns
        if daily is None:
            daily = market["spot_avg"].pct_change().reindex(X.index)
        cv = PurgedWalkForward(n_splits=6, min_train=max(400, len(X) // 4))
        results[name] = evaluate_stream(
            name, calibrated_fold_proba(X, y, cv), y, daily, ledger
        )

    print(f"\n=== EDGE HUNT ROUND 5: GUATEMALA SUPPLY [{tag}] — tranched 5d ===")
    print("    champion: physics-gbm +0.79 (DSR 0.63 vs 32)\n")
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
    print(f"  Ledger grew {len(ledger) - 2} -> {len(ledger)}.")
    print("  Rule: the feed stays only if it pays; the champion changes only "
          "if beaten with the haircut counted.")


if __name__ == "__main__":
    main()
