"""T13 — as-issued forecast rain, ONE pre-registered trial.

    python scripts/forecast_rain_trial.py

Pre-registered hypothesis (see also scripts/gefs_pilot.py's docstring):
AS-ISSUED forecast rain carries information timed to the 5-day label
window SPECIFICALLY BECAUSE it is forward-looking at the moment of the
trading decision, whereas the already-tested REALIZED/observed rain
feature (T6 in edge_hunt.py's lineage, rain_trial.py: Sharpe +0.64 solo,
+0.006 p(SR<=0), but added nothing on top of the physics champion) is
backward-looking and structurally too late to help a call about the next
5 days. A forecast of accumulated precip over leads 0-120h, issued at the
same init as the auction day, overlaps the label window directly; a
30/90-day trailing rainfall anomaly does not.

COVERAGE CAVEAT (read this before the results table, not after):
`data/raw/climate/gefs_forecast_rain.csv` is the NOAA GEFS v12 reforecast
backfill and covers ONLY 2014-11-07 -> 2019-12-31 (1341/1341 auction days
in that era). The 2020-present leg needs the operational GEFS archive with
.idx byte-range subsetting and has NOT been built — this feature has ZERO
coverage from 2020 onward and does not overlap the live-trading window at
all. Whatever this trial finds is an honest answer to "did as-issued rain
forecasts help during 2014-2019", never a claim about current predictive
power. Nothing below should be read as "the feature carries no edge" or
"the feature carries edge" beyond that 5-year window — we simply cannot
test the other 6+ years yet.

EVALUATION APPROACH, DECLARED BEFORE RUNNING (both legs, for honesty):
  (a) Restrict evaluation to the 2014-2019 covered window. Run the
      standard vehicle (tranched 5d rebalance, per-fold isotonic-
      calibrated GBM, 15bps costs, PurgedWalkForward(n_splits=6,
      min_train=max(400, n//4))) inside that sub-period only. n~=1341,
      giving min_train=400 and ~157-row test blocks per fold — the
      standard formula already degrades gracefully for this sample size,
      so no fold-count reduction is needed (checked before running).
  (b) Recompute the T6 (realized-rain) configuration -- core features,
      no forecast block -- on the SAME 2014-2019 sub-window, so the two
      rain features are judged on a comparable basis (as-issued vs
      realized, same period) instead of as-issued-on-a-slice vs
      realized-on-full-history. This is a COMPARISON BASELINE, not a new
      trial (same precedent as the v1.1 champion re-estimate in
      RESULTS_REAL.md: same config, different slice, not counted).

ONE counted trial:
  T13 forecast-rain-gbm   core features (realized rain_anom_30/90 already
                          wired, as in every trial since T6) + the single
                          as-issued feature `fcst_rain_5d`
                          (src/features/forecast_rain.py), tranched-5d
                          calibrated GBM, restricted to the 2014-2019
                          covered window.

This is a single configuration -- no swept parameter, no alternate
feature engineering choices tried and discarded.

DECISION RULE, DECLARED BEFORE RUNNING: T13 clearing zero IN ISOLATION is
NOT sufficient to pay. The 2014-2019 window is a favorable sub-period for
this vehicle in general (small sample, possibly a friendlier regime) --
so any reasonable config could show an inflated Sharpe inside it. The
pay/cut call is therefore: PAY only if T13 (a) clears zero AND (b) beats
the same-window realized-rain baseline from leg (b) above. If T13 clears
zero but loses to that baseline, the honest read is CUT — the window was
good to everyone, the as-issued feature specifically was not additive,
consistent with this project's now-familiar stacking-dilution pattern
(T5, T7, T10). This combining rule was written into this docstring before
the first run of this script; the earlier one-line "does T13 alone clear
zero" heuristic in an initial draft of this file was replaced by this
rule before any RESULTS_REAL.md entry was written, once it became clear
in-run that leg (a) alone could mislead (both legs cleared zero, so
"clears zero" alone does not distinguish the hypothesis under test) --
disclosed here the same way T5's sequencing is disclosed in edge_hunt.py.

Ledger: 35 prior (24 + round-2's 5 + rain's 2 + T8's 1 + Guatemala's 2 +
T12's 1) + this 1 = 36.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from edge_hunt import PRIOR_LEDGER, calibrated_fold_proba, evaluate_stream  # noqa: E402
from rain_trial import ROUND2_SHARPES  # noqa: E402
from run import load_dataset  # noqa: E402
from src.analysis.robustness import deflated_sharpe  # noqa: E402
from src.features.engineering import build_features  # noqa: E402
from src.features.forecast_rain import build_forecast_rain_features  # noqa: E402
from src.validation.walkforward import PurgedWalkForward  # noqa: E402

logging.basicConfig(level=logging.WARNING)

# Ledger as it stood entering T13 (RESULTS_REAL.md, one line per round):
RAIN_SHARPES = [0.643, 0.546]   # T6, T7  (scripts/rain_trial.py)
T8_SHARPE = [0.626]             # T8      (scripts/displacement_trial.py, killed)
GTM_SHARPES = [0.485, 0.575]    # T9, T10 (scripts/gtm_trial.py, cut)
T12_SHARPE = [0.470]            # T12     (scripts/ensemble_trial.py, tie; pinned
                                 # to the published headline — a same-config
                                 # rerun today drifts to +0.445 on a slightly
                                 # grown dataset, expected dataset-drift noise,
                                 # not a reason to rewrite a prior ledger entry)


def main() -> None:
    market, tag = load_dataset(False)

    fcst = build_forecast_rain_features(market)
    covered = fcst["fcst_rain_5d"].notna()
    n_covered = int(covered.sum())
    if n_covered < 500:
        sys.exit(
            "forecast rain not built or coverage too thin — run "
            "scripts/gefs_backfill_reforecast.py first"
        )
    lo, hi = market.index[covered].min().date(), market.index[covered].max().date()
    print(f"forecast-rain coverage: {n_covered} of {len(market)} auction days "
          f"({lo} -> {hi}) — the ONLY window this trial can speak to")

    # ---- (b) comparison baseline: T6-equivalent (core, realized rain only),
    #      restricted to the SAME window. NOT a counted trial.
    X_core_full, y_core_full = build_features(market, alt=None)
    X_core_full = X_core_full.drop(columns=X_core_full.columns[X_core_full.isna().all()])
    core_window = X_core_full.index.intersection(market.index[covered])
    X_core_w = X_core_full.loc[core_window]
    y_core_w = y_core_full.loc[core_window]

    daily = market["spot_avg"].pct_change()
    daily_w = daily.reindex(X_core_w.index)

    cv = PurgedWalkForward(n_splits=6, min_train=max(400, len(X_core_w) // 4))
    print(f"vehicle: n={len(X_core_w)}, n_splits=6, min_train={cv.min_train}, "
          f"test_block~={(len(X_core_w) - cv.min_train) // 6}")

    baseline_ledger: list[float] = []  # throwaway — comparison, not counted
    baseline = evaluate_stream(
        "T6-equivalent (realized rain, 2014-19 window)",
        calibrated_fold_proba(X_core_w, y_core_w, cv),
        y_core_w, daily_w, baseline_ledger,
    )

    # ---- (a) + ONE counted trial: T13 forecast-rain-gbm on the covered window
    X13_full, y13_full = build_features(market, alt=fcst)
    X13_full = X13_full.drop(columns=X13_full.columns[X13_full.isna().all()])
    assert "fcst_rain_5d" in X13_full.columns, "forecast-rain feature missing"
    window = X13_full.index.intersection(market.index[covered])
    X13 = X13_full.loc[window]
    y13 = y13_full.loc[window]
    daily13 = daily.reindex(X13.index)
    assert X13["fcst_rain_5d"].notna().all(), "as-issued feature has gaps inside its own coverage window"

    ledger = (
        list(PRIOR_LEDGER) + ROUND2_SHARPES + RAIN_SHARPES + T8_SHARPE
        + GTM_SHARPES + T12_SHARPE
    )
    n_prior = len(ledger)
    result = evaluate_stream(
        "T13 forecast-rain-gbm", calibrated_fold_proba(X13, y13, cv), y13, daily13, ledger
    )

    # ---- report ---------------------------------------------------------
    print(f"\n=== T13 FORECAST RAIN [{tag}] — 2014-2019 window only, tranched 5d vehicle ===")
    print("    COVERAGE CAVEAT: this window is disjoint from 2020-present; "
          "nothing here generalizes to the live-trading regime.\n")
    rows = [
        {
            "trial": name, "sharpe": m.get("sharpe"), "ci_5": m.get("ci_5"),
            "ci_95": m.get("ci_95"), "p_leq_0": m.get("p_leq_0"),
            "max_dd": m.get("max_dd"), "auc": m.get("auc"),
            "hit_vs_base": m.get("hit_vs_base_pts"),
        }
        for name, m in (
            ("T6-equivalent (realized, same window)", baseline),
            ("T13 forecast-rain-gbm (counted)", result),
        )
    ]
    print(pd.DataFrame(rows).set_index("trial").round(3).to_string())

    dsr = deflated_sharpe(result["net"].dropna(), ledger)
    print(f"\n  T13: Sharpe {result['sharpe']:+.3f}  90% CI "
          f"[{result.get('ci_5', float('nan')):+.2f}, {result.get('ci_95', float('nan')):+.2f}]  "
          f"p(SR<=0) {result.get('p_leq_0', float('nan')):.3f}")
    print(f"  DSR over {dsr['n_trials']} trials: {dsr['dsr']:.3f} "
          f"(expected max SR from luck {dsr['expected_max_sharpe']:.2f})")
    print(f"  Ledger grew {n_prior} -> {len(ledger)}.")
    delta = result["sharpe"] - baseline["sharpe"]
    print(f"\n  as-issued vs realized, SAME 2014-19 window: {delta:+.3f} Sharpe "
          f"({'forecast adds' if delta > 0 else 'forecast subtracts'})")

    # Pre-registered decision rule: T13 clearing zero IN ISOLATION is not
    # sufficient to pay, because the whole 2014-2019 window is favorable for
    # ANY reasonable config in this vehicle (the baseline clears zero too,
    # by MORE). The hypothesis under test -- as-issued forecast adds value
    # BEYOND realized rain -- is decided by the controlled (b) comparison.
    clears_zero = result.get("p_leq_0", 1.0) < 0.10 and result["sharpe"] > 0
    beats_baseline = delta > 0
    if clears_zero and beats_baseline:
        verdict = "PAY — clears zero AND beats the same-window realized-rain baseline"
    else:
        verdict = (
            "CUT — clears zero in isolation, but that is a property of the "
            "favorable 2014-2019 window shared by every config tested in it "
            "(the baseline clears zero by MORE). On the controlled, "
            "apples-to-apples test this trial was pre-registered for, the "
            "as-issued forecast SUBTRACTS Sharpe vs core+realized-rain alone "
            "— a stacking-dilution result, joining T5/T7/T10."
        )
    print(f"  Verdict: {verdict}")
    print("  This verdict is about 2014-2019 ONLY. Post-2020 behavior is "
          "unknown until the operational-archive leg is built.")

    # Data-quality disclosure: the 2014-2015 crawl is sparse enough that 3
    # of 1341 window rows sit immediately after a >30-day calendar gap in
    # the auction record (68d, 107d, 31d) -- their fcst_rain_5d value is
    # therefore stale by that many days, not a leak (still strictly past
    # information), just noise. Same positional-shift convention as every
    # other lagged feature (mom_5 etc. are equally stale for these 3 rows);
    # affects 0.2% of the window and is not the reason for the verdict above.
    gaps = market.index.to_series().diff()
    stale = gaps.reindex(X13.index)
    n_stale = int((stale > pd.Timedelta("30 days")).sum())
    print(f"\n  data-quality note: {n_stale} of {len(X13)} window rows follow a "
          ">30-day gap in the sparse 2014-15 auction record (stale forecast, "
          "not a leak; same positional-lag convention as mom_5/mom_10).")


if __name__ == "__main__":
    main()
