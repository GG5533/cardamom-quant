"""Edge hunt, round 6 — the probability ensemble, ONE pre-registered trial.

    python scripts/ensemble_trial.py

Feature STACKING diluted four times (T5, T7, T10, and the original alt
block): giving one tree more columns makes it worse on ~500 effective
bets. But three configurations cleared zero SOLO — physics (+0.79 single-
layout / +0.43 slicing-averaged), kalman (+0.67), rain (+0.64) — each
seeing the market through a different mechanism. Averaging their
CALIBRATED PROBABILITIES (bagging) is a different operation from feature
stacking: variance of the blend falls with decorrelation while each
member keeps its own bias discipline.

  T12 ensemble-gbm   per-day mean of the three calibrated streams:
                       - physics: pre-rain core + auction physics
                       - kalman:  pre-rain core + fold-fitted anomaly
                       - rain:    core with the IMD rain features live
                     evaluated SLICING-ROBUST from the start: the same
                     declared 6-layout family as robust_estimate.py,
                     per-layout ensemble Sharpes reported, the blended
                     stream is the trial's number.

One trial; ledger 34 -> 35. If it does not beat the champion's standing
+0.43 blended estimate, it is a kill and the backtest modeling front
closes until live/basis/forecast-rain evidence arrives.
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

from edge_hunt import PRIOR_LEDGER, calibrated_fold_proba  # noqa: E402
from horizon_experiment import tranched_net_returns  # noqa: E402
from rain_trial import ROUND2_SHARPES  # noqa: E402
from run import load_dataset  # noqa: E402
from src.analysis.robustness import block_bootstrap_sharpe, deflated_sharpe  # noqa: E402
from src.features.auction_physics import build_physics_features  # noqa: E402
from src.features.engineering import build_features  # noqa: E402
from src.metrics import backtest_metrics, classification_metrics  # noqa: E402
from src.models.kalman_seasonal import kalman_features  # noqa: E402
from src.validation.walkforward import PurgedWalkForward  # noqa: E402

logging.basicConfig(level=logging.WARNING)

LAYOUTS = [(s, off) for s in (5, 6, 7) for off in (0, 63)]
RAIN_SHARPES = [0.643, 0.546]
T8_SHARPE = [0.626]
GTM_SHARPES = [0.485, 0.575]
CHAMPION_BLENDED = 0.431


def member_matrices(market_full: pd.DataFrame):
    """The three member designs: (X, y) each, on their own market variant."""
    pre = market_full.drop(columns=["rain_mm", "rain_climatology", "rain_anomaly"])
    out = {}
    X, y = build_features(pre, alt=build_physics_features(pre))
    out["physics"] = (X.drop(columns=X.columns[X.isna().all()]), y, pre)
    out["kalman"] = ("fold-fitted", y, pre)  # built per fold below
    Xr, yr = build_features(market_full, alt=None)
    out["rain"] = (Xr.drop(columns=Xr.columns[Xr.isna().all()]), yr, market_full)
    return out


def main() -> None:
    market, tag = load_dataset(False)
    members = member_matrices(market)
    y = members["physics"][1]
    pre = members["physics"][2]
    px = pre["spot_avg"].reindex(members["physics"][0].index)
    daily = pre["spot_avg"].pct_change().reindex(members["physics"][0].index)
    X_core = members["physics"][0][  # core columns only, for the kalman member
        [c for c in members["physics"][0].columns
         if c not in ("bid_dispersion", "dispersion_z_63", "inv_overhang", "hurst_63")]
    ]
    base_min = max(400, len(y) // 4)

    layout_streams, rows = [], []
    for n_splits, off in LAYOUTS:
        cv = PurgedWalkForward(n_splits=n_splits, min_train=base_min + off)
        streams = []
        # physics + rain: static feature matrices
        streams.append(calibrated_fold_proba(members["physics"][0], y, cv))
        streams.append(calibrated_fold_proba(
            members["rain"][0], members["rain"][1], cv).reindex(y.index))

        def x_kalman(train_end: int) -> pd.DataFrame:
            feats, _ = kalman_features(px, train_end)
            return X_core.join(feats)

        streams.append(calibrated_fold_proba(x_kalman, y, cv))

        ens = pd.concat(streams, axis=1).mean(axis=1)
        mask = ens.notna()
        net, _ = tranched_net_returns(ens[mask], daily[mask])
        m = backtest_metrics(net)
        rows.append({"layout": f"{n_splits} folds · +{off}", "sharpe": m["sharpe"]})
        layout_streams.append(ens)

    per = pd.DataFrame(rows).set_index("layout")
    print(f"\n=== T12 PROBABILITY ENSEMBLE [{tag}] — slicing-robust ===")
    print(per.round(3).to_string())
    print(f"  layout mean {per['sharpe'].mean():+.3f}  "
          f"range [{per['sharpe'].min():+.2f}, {per['sharpe'].max():+.2f}]")

    blend = pd.concat(layout_streams, axis=1).mean(axis=1)
    mask = blend.notna()
    net, _ = tranched_net_returns(blend[mask], daily[mask])
    m = backtest_metrics(net)
    m.update(block_bootstrap_sharpe(net))
    m.update(classification_metrics(y[mask], blend[mask]))

    ledger = (list(PRIOR_LEDGER) + ROUND2_SHARPES + RAIN_SHARPES
              + T8_SHARPE + GTM_SHARPES + [m["sharpe"]])
    dsr = deflated_sharpe(net.dropna(), ledger)
    print(f"\n  T12 blended: Sharpe {m['sharpe']:+.3f}  "
          f"90% CI [{m['ci_5']:+.2f}, {m['ci_95']:+.2f}]  p(SR<=0) {m['p_leq_0']:.3f}")
    print(f"  AUC {m['auc']:.3f}  hit vs base {m['hit_vs_base_pts']:+.1f}pts  "
          f"max_dd {m['max_dd']:+.1%}")
    print(f"  vs champion blended {CHAMPION_BLENDED:+.2f}  |  "
          f"DSR over {dsr['n_trials']} trials: {dsr['dsr']:.3f}")
    print(f"  Ledger grew {len(ledger) - 1} -> {len(ledger)}.")
    verdict = "PROMOTE — new standing champion" \
        if m["sharpe"] > CHAMPION_BLENDED else "KILL — champion stands"
    print(f"  Verdict: {verdict}")


if __name__ == "__main__":
    main()
