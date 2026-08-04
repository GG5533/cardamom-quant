"""MCX futures basis, ONE pre-registered trial — T14.

    python scripts/mcx_basis_trial.py

Hypothesis (classic commodity basis / convenience-yield signal): the spread
between the MCX front-month futures price and the spot auction price,
basis_pct = (fut_close - spot_avg) / spot_avg, carries information about
forward spot direction — backwardation (futures below spot, high
convenience yield / near-term scarcity) and contango should tilt the odds
differently. This is the first trial to touch MCX data: the future was
suspended 2021 -> relaunched 2025-07-29, and the relaunch Bhavcopy backfill
(data/raw/mcx/CommodityWise_CARDAMOM_backfill.csv, 1305 contract-day rows)
was only just dropped in and parsed by the existing MCXBhavcopyLoader
(src/data/mcx_bhavcopy.py, unchanged) into a continuous front-month series.
`scripts/merge_mcx.py` wires it into market.parquet's basis/basis_pct
columns, which build_features() (src/features/engineering.py) already had
declared in CORE_FEATURE_COLS and already shifted causally
(.shift(1) / .diff(5).shift(1)) — the columns were dark, not absent.

THE COVERAGE PROBLEM, faced head-on before running anything for results:
basis_pct is non-null for 232 of 3153 feature rows (7.4%) -- 2025-07-30
through 2026-07-11 -- inside an 11-year dataset (2014-11-07 onward). It is
NaN for the ENTIRE pre-relaunch history and for the whole 2021-2025
suspension gap, by construction.

Checked analytically before any model touched results: under the
project's standard vehicle (PurgedWalkForward n_splits=6,
min_train=max(400, n//4)=788 on the champion's n=3153), every one of the
6 folds' TRAINING window ends at or before row ~2748-2983 -- before row
2873, the first row with non-null basis_pct. That means a naive
full-history run would fit a GBM that NEVER sees a single non-missing
basis_pct/basis_chg value in ANY fold's training set; the feature is
structurally inert there, not just weak. This is not hypothetical --
attempting exactly that run (champion + basis on the standard 6-fold
vehicle, reusing scripts/ensemble_trial.py's own "physics" member matrix
unmodified) throws inside sklearn's HistGradientBoostingClassifier
binning step (`ValueError: window shape cannot be larger than input array
shape` -- ` _find_binning_thresholds` needs >=2 distinct non-missing
values per feature and gets zero). Reported below as a diagnostic, NOT a
counted trial: the naive full-history vehicle doesn't produce a
misleading number, it doesn't run at all, which is itself the honest
disclosure the task asked for.

DESIGN CHOSEN -- option (a), literally: restrict the entire train +
calibrate + test universe to the covered window (the standard vehicle's
calibration slice alone reserves ~20% of an expanding multi-thousand-row
training set, which already exceeds the whole 232-day covered sample --
there is no way to nest a small covered-only test tail inside the
standard vehicle's fold geometry and still leave the calibration-and-fit
slices any real exposure; tried, confirmed by direct computation before
picking this design, see git history of this file for the rejected
nested-burn-in attempt). Concretely:

  * universe: the 281 feature rows with date >= 2025-07-29 (the relaunch),
    built from the FULL market frame so momentum/vol/physics features keep
    their full-history rolling windows -- only the row SUBSET used for
    CV is restricted, not the feature computation itself;
  * PurgedWalkForward(n_splits=3, min_train=140, purge=5, embargo=5) --
    min_train=140 is roughly half the covered universe, derived from
    "leaves >=75 non-calibration training rows after the pipeline's own
    max(50, 20%-of-train) calibration carve-out" (checked: fold0 inner-fit
    62/75 rows have non-null basis, fold1 101/122, fold2 139/169) --
    not swept for a better number, chosen once from that arithmetic;
    n_splits=3 (not 1 or 2) to avoid a single-anchor estimate, the
    project's own documented lesson (RESULTS_REAL.md: anchor luck is
    worth +/-0.4 Sharpe here) -- three 47-day test blocks, 141 OOS days
    total, 2026-01-20 -> 2026-07-13;
  * everything else matches the standard vehicle exactly: per-fold
    isotonic-calibrated GBM probabilities (calibrated_fold_proba),
    Jegadeesh-Titman tranched 5d rebalancing (tranched_net_returns), 15bps
    costs, block-bootstrap 90% CI, DSR against the full ledger.

  T14 mcx-basis-gbm   champion (pre-rain core + auction-physics block --
                      the T1/T8/T9/T10/T12 configuration) + basis_pct +
                      basis_chg, evaluated on the covered-window folds
                      above.

  Control (NOT a separate counted trial -- same precedent as
  robust_estimate.py's re-slicing of the already-ledgered champion across
  6 fold layouts, logged there as "estimator refinement, not selection;
  the DSR ledger is unchanged"): the identical champion WITHOUT basis,
  evaluated on the IDENTICAL covered-window folds, as T14's ablation
  baseline. This is the paired control the hypothesis needs, not a second
  hypothesis competing for promotion.

141 OOS days is a small, likely underpowered sample -- ~28 non-overlapping
5-day bets before tranching smooths the count up a little. The 90%
bootstrap CI and DSR below are expected to be wide, and that width IS the
honest answer, not a defect to explain away. If the honest reading is "we
cannot say anything meaningful yet, the sample is too short since
relaunch," that is a fully successful trial outcome, written up with the
same care as any clean kill (see T3/T4, edge_hunt.py).

Ledger: 35 prior (24 + round-2's 5 + rain's 2 + T8 + Guatemala's 2 + T12) +
this 1 = 36. T13 is reserved for the forecast-rain trial (not yet run,
scripts/gefs_pilot.py) and is deliberately skipped here.
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

from edge_hunt import PRIOR_LEDGER, calibrated_fold_proba, evaluate_stream  # noqa: E402
from rain_trial import ROUND2_SHARPES  # noqa: E402
from run import load_dataset  # noqa: E402
from src.analysis.robustness import deflated_sharpe  # noqa: E402
from src.features.auction_physics import build_physics_features  # noqa: E402
from src.features.engineering import build_features  # noqa: E402
from src.models.baselines import make_gbm  # noqa: E402
from src.validation.walkforward import PurgedWalkForward  # noqa: E402

logging.basicConfig(level=logging.WARNING)

RAIN_SHARPES = [0.643, 0.546]     # T6, T7 (scripts/rain_trial.py output)
T8_SHARPE = [0.626]               # displacement kill (scripts/displacement_trial.py)
GTM_SHARPES = [0.485, 0.575]      # T9, T10 (scripts/gtm_trial.py output)
T12_SHARPE = [0.470]              # ensemble, blended (RESULTS_REAL.md, round 6)

MIN_TRAIN = 140
N_SPLITS = 3


def _naive_full_history_diagnostic(pre: pd.DataFrame) -> None:
    """Confirm (not count) that the standard 6-fold vehicle cannot learn
    from basis at all: every fold's training set is 100% NaN for it.
    """
    phys = build_physics_features(pre)
    X, y = build_features(pre, alt=phys)
    X = X.drop(columns=X.columns[X.isna().all()])
    assert "basis_pct" in X.columns, "merge_mcx.py has not been run"
    cv = PurgedWalkForward(n_splits=6, min_train=max(400, len(X) // 4))
    first_covered = X.index.get_loc(X.index[X["basis_pct"].notna()][0])
    dead_folds = 0
    for tr, _te in cv.split(len(X)):
        if X["basis_pct"].iloc[tr].notna().sum() == 0:
            dead_folds += 1
    print(f"  diagnostic: first non-null basis_pct at row {first_covered} of "
          f"{len(X)}; {dead_folds}/6 standard-vehicle folds have ZERO "
          f"non-missing basis_pct in their training set.")
    try:
        calibrated_fold_proba(X, y, cv)
        print("  (unexpectedly ran to completion -- see writeup, treat with "
              "suspicion)")
    except ValueError as e:
        print(f"  confirmed: the naive full-history fit raises -- {e!r}")


def main() -> None:
    market, tag = load_dataset(False)
    pre = market.drop(columns=["rain_mm", "rain_climatology", "rain_anomaly"])

    print(f"=== T14 MCX BASIS [{tag}] — covered-window vehicle ===\n")
    _naive_full_history_diagnostic(pre)

    phys = build_physics_features(pre)
    X_full, y_full = build_features(pre, alt=phys)
    X_full = X_full.drop(columns=X_full.columns[X_full.isna().all()])
    assert {"basis_pct", "basis_chg"} <= set(X_full.columns)

    covered = X_full.index >= "2025-07-29"
    n_cov = int(covered.sum())
    print(f"\n  covered universe: {n_cov} rows, "
          f"{X_full.index[covered].min().date()} -> "
          f"{X_full.index[covered].max().date()}, "
          f"{int(X_full.loc[covered, 'basis_pct'].notna().sum())} with "
          "non-null basis_pct")

    Xc = X_full.loc[covered]
    yc = y_full.loc[covered]
    daily = market["spot_avg"].pct_change().reindex(Xc.index)
    cv = PurgedWalkForward(n_splits=N_SPLITS, min_train=MIN_TRAIN, purge=5, embargo=5)

    ledger = list(PRIOR_LEDGER) + ROUND2_SHARPES + RAIN_SHARPES + T8_SHARPE + GTM_SHARPES + T12_SHARPE
    control_ledger = list(ledger)  # throwaway copy; control is not counted

    X_base = Xc.drop(columns=["basis_pct", "basis_chg"])
    control = evaluate_stream(
        "control champion (no basis, covered window)",
        calibrated_fold_proba(X_base, yc, cv), yc, daily, control_ledger,
    )
    t14 = evaluate_stream(
        "T14 mcx-basis-gbm", calibrated_fold_proba(Xc, yc, cv), yc, daily, ledger
    )

    rows = []
    for name, m in (("control (no basis)", control), ("T14 mcx-basis-gbm", t14)):
        rows.append({
            "config": name, "sharpe": m.get("sharpe"), "ci_5": m.get("ci_5"),
            "ci_95": m.get("ci_95"), "p_leq_0": m.get("p_leq_0"),
            "max_dd": m.get("max_dd"), "auc": m.get("auc"),
            "hit_vs_base": m.get("hit_vs_base_pts"), "n_oos": int(m.get("net").notna().sum()),
        })
    print("\n" + pd.DataFrame(rows).set_index("config").round(3).to_string())

    dsr = deflated_sharpe(t14["net"].dropna(), ledger)
    print(f"\n  T14: Sharpe {t14['sharpe']:+.3f}  90% CI "
          f"[{t14.get('ci_5', float('nan')):+.2f}, {t14.get('ci_95', float('nan')):+.2f}]  "
          f"p(SR<=0) {t14.get('p_leq_0', float('nan')):.3f}")
    print(f"  DSR over {dsr['n_trials']} trials: {dsr['dsr']:.3f} "
          f"(expected max SR from luck {dsr['expected_max_sharpe']:.2f})")
    print(f"  delta vs control (same folds): "
          f"{t14['sharpe'] - control['sharpe']:+.3f} Sharpe")
    print(f"\n  Ledger grew {len(ledger) - 1} -> {len(ledger)}.")
    print("  Sample is small by design (141 OOS days, ~28 non-overlapping "
          "bets) -- the covered window is all there is since relaunch.")


if __name__ == "__main__":
    main()
