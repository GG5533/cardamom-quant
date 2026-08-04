# REAL out-of-sample results — July 7, 2026

**Data:** 5,655 Spices Board e-auction sessions → 3,148 auction days,
07-Nov-2014 → 07-Jul-2026, crawled live via browser on 07-Jul-2026 (15
malformed source rows quarantined, never imputed — see
`data/raw/browser/sessions_repaired.csv.rejected`). Prices ₹495–4,675/kg;
the 2019 spike to ₹4,675 (post-2018-Kerala-flood shock) shows up exactly
where history says it should — a free sanity check on the whole pipeline.
Plus real NOAA ONI (2010→Feb-2026, publication-lagged). Rain, basis, FX and
Comtrade feeds not yet wired into this run.

## The scorecard (purged walk-forward, 6 folds, after 15bps costs)

| trial | hit vs base | AUC | Sharpe | 90% bootstrap CI | p(SR≤0) |
|---|---|---|---|---|---|
| seasonal baseline | +3.05 pts | 0.531 | **+0.30** | [−0.28, +0.85] | 0.20 |
| core / logistic | +1.27 pts | 0.531 | −0.20 | [−0.68, +0.26] | 0.78 |
| core / gbm | **+4.16 pts** | **0.553** | +0.20 | [−0.29, +0.69] | 0.23 |
| alt / logistic | +0.93 pts | 0.496 | −0.41 | [−0.87, +0.07] | 0.93 |
| alt / gbm | +2.08 pts | 0.537 | −0.18 | [−0.66, +0.28] | 0.74 |

Best Sharpe (seasonal baseline): PSR vs 0 = 0.821; after the multiple-testing
haircut for all 6 trials, **DSR = 0.383**.

## The findings — reported as found

1. **No deployable edge after costs.** Nothing has a Sharpe distinguishable
   from zero (every bootstrap CI straddles it; the best DSR is 0.38).
2. **Predictive signal exists but doesn't monetize.** Core GBM beats the
   base rate by +4.2pts with AUC 0.553 — real classification skill — yet
   daily-rebalanced conviction sizing hands it back in turnover costs.
   The obvious next experiment: trade only high-conviction signals
   (|p−0.5| threshold) or rebalance at the 5-day label horizon.
3. **The seasonal rule is the one to beat, and it wasn't beaten.** Long
   lean-season earns Sharpe 0.84 inside Mar–Jul and gives back −0.43 the
   rest of the year. Exactly the regime concentration the regime table is
   for.
4. **Complexity hurt, twice.** Adding the alt-feature block degraded both
   ML models on real data (gbm +4.16 → +2.08pts). The synthetic testbed's
   "regularized-linear beats booster" lesson generalized into "on 3,000
   noisy samples, every added feature costs more than it pays."
5. **GBM probabilities are miscalibrated** (Brier 0.270 vs climatology
   0.250, max bin gap +0.27) — conviction sizing on uncalibrated
   probabilities compounds the cost problem. Platt/isotonic calibration is
   a cheap next step.

## UPDATE 07-Jul-2026 (local): horizon-matched trading — the "maybe"

Finding 2's next experiment has been run (`scripts/horizon_experiment.py`):
core/gbm probabilities, raw and per-fold isotonic-calibrated (fit on a
purged tail slice of each training window), traded through a grid of
rebalance cadence ∈ {1d, 5d} × conviction threshold ∈ {0, .05, .10, .15}.
Every cell is a counted trial; the ledger below reflects all of them.

Calibration first (it feeds 2p−1 sizing, so it is a sizing fix): OOS Brier
0.2667 → 0.2541 (climatology 0.2500), max bin gap +0.234 → +0.175.

| trial (core/gbm) | Sharpe | 90% CI | p(SR≤0) | ann. turnover |
|---|---|---|---|---|
| raw / 1d / thr 0 (= published) | +0.12 | [−0.38, +0.64] | 0.33 | 16.6× |
| raw / 5d / thr 0 | +0.77 | [+0.32, +1.24] | 0.006 | 6.0× |
| calibrated / 5d / thr 0 | +0.93 | [+0.46, +1.41] | 0.001 | 2.8× |
| …13 more cells in the script output | | | | |

Five of sixteen cells clear zero at 90% — all five at 5d rebalancing, none
at 1d. Conviction thresholds only shave turnover; they never add Sharpe once
the cadence is horizon-matched. Turnover, not skill, was the whole story.

**The anchor-day catch (reported because it bites):** a single 5d-rebalance
backtest silently picks 1 of 5 possible anchor days. Sweeping the anchor:
raw {+0.77, +0.07, +0.31, +0.28, +0.63}, calibrated {+0.93, +0.24, +0.20,
+0.62, +0.49}. All ten positive, but the headline cells above are the
luckiest anchors. The deliverable is therefore the anchor-free **tranched
book** (1/5 of capital rebalances each day, Jegadeesh-Titman overlap fix;
costs paid per tranche, no internal netting — conservative):

| trial | Sharpe | 90% CI | p(SR≤0) | MaxDD | DSR (all trials) |
|---|---|---|---|---|---|
| tranched 5d / raw | +0.48 | [+0.00, +0.96] | 0.049 | −8.7% | 0.38 (23 trials) |
| tranched 5d / calibrated | **+0.57** | [+0.05, +1.12] | 0.036 | −9.6% | **0.49** (24 trials) |

**Trial ledger: 6 (original scorecard) + 16 (grid) + 2 (tranched) = 24.**
Expected max Sharpe from luck alone across 24 zero-skill trials: 0.58.

### Updated verdict

The calibrated, horizon-matched, anchor-free book beats the seasonal
baseline (+0.57 vs +0.30), its bootstrap CI clears zero, and p(SR≤0) is
3.6% — the first configuration in this project to manage any of that. But
DSR = 0.49 against the 24-trial search: the result does **not** decisively
survive the multiple-testing haircut. Honest label: *a defensible maybe,
not a discovery.* What would settle it: the unwired feeds paying OOS, or
simply more history (the CI narrows as √T with live auction days).

## UPDATE (edge hunt, round 2): auction physics is the new headline

Five trials pre-registered in `scripts/edge_hunt.py` (T5's sequencing
disclosed there), all in the tranched-5d calibrated vehicle, all counted:

| trial | ingredient | Sharpe | 90% CI | p(SR≤0) |
|---|---|---|---|---|
| **T1 physics-gbm** | bid dispersion + inventory overhang + Hurst | **+0.79** | [+0.28, +1.29] | **0.006** |
| T2 kalman-gbm | causal Kalman anomaly features | +0.67 | [+0.11, +1.21] | 0.025 |
| T3 ou-bands | OU first-passage band policy, standalone | −0.22 | [−0.73, +0.28] | 0.78 |
| T4 conformal-gbm | split-conformal gate on a 5d regressor | −0.17 | [−0.51, +0.09] | 0.87 |
| T5 physics+kalman | both feature blocks | +0.71 | [+0.20, +1.20] | 0.012 |

What the round taught, reported as found:

1. **The unexploited auction columns carried real signal.** `spot_max /
   spot_avg` (competition intensity at the auction), crop-year inventory
   overhang (Kaldor–Working storage state, past-years-only norm), and a
   rolling Hurst regime dial lift the calibrated tranched book from +0.57
   to **+0.79** (hit vs base +5.6pts, AUC 0.555) with max drawdown −7.5%.
2. **The Kalman anomaly helps alone (+0.67) but adds nothing on top of
   physics** (T5 +0.71 < T1 +0.79) — the tree was already finding the
   reversion through the physics block.
3. **Two clean kills.** The standalone OU band policy dies because the
   anomaly's fitted half-life is ~138 days — far too slow to band-trade
   at 15bps (and unit-weight entries produced a −92% drawdown path). The
   conformal gate throttled the signal to nothing (AUC 0.499). Both stay
   in the ledger.
4. **Deflated Sharpe: 0.63** for T1 against all **29** trials ever run
   (expected max from luck alone: 0.68). Better than 0.49, still short of
   discovery grade. The verdict upgrades from "defensible maybe" to
   "strengthening maybe" — nothing more.

## UPDATE (edge hunt, round 3): the rain feed is wired — and it pays as
## signal, not as the champion

IMD gridded rainfall (2010–2025, Idukki box) downloaded, validated and
merged (`scripts/merge_rain.py`; 2,997 of 3,148 auction days covered, 2026
NaN — the feed's real publication lag). Free natural-experiment check: the
largest anomaly in 16 years is **+164 mm/day on 2018-08-16 — the Kerala
flood peak**, exactly where STEP1's acceptance criterion asked for it.

Two pre-registered trials (`scripts/rain_trial.py`), ledger 29 → 31:

| trial | Sharpe | 90% CI | p(SR≤0) | AUC | hit vs base |
|---|---|---|---|---|---|
| T6 rain-gbm | +0.64 | [+0.06, +1.24] | 0.039 | **0.572** | **+6.3pts** |
| T7 physics+rain-gbm | +0.55 | [−0.03, +1.14] | 0.056 | 0.560 | +4.4pts |

Read honestly: the weather thesis validates *predictively* — rain gives
the best classification numbers of the whole project (AUC 0.572, +6.3pts)
and clears zero on its own — but it does not beat the physics champion in
the money metric, and stacking blocks dilutes again (third time: T5, T7).
On ~500 effective independent 5d bets, more features ≠ more Sharpe.

**Standing champion, re-verified against the full 31-trial ledger:**
physics-gbm Sharpe **+0.79**, DSR **0.63** (expected max from luck 0.68).
The rain feed STAYS (it pays solo; the data is real and wired); the
champion configuration is unchanged. Verdict remains *strengthening
maybe* — DSR must clear ~0.9 before anyone gets to say edge.

## UPDATE (edge hunt, round 4): first displacement trial — killed

T8 (`scripts/displacement_trial.py`, sequencing disclosed) dropped
mom_10/mom_63 from the champion on the variance-reduction hypothesis the
last-fold dAUC diagnostics suggested. Result: Sharpe +0.63 [+0.13, +1.12],
AUC 0.549, hit +3.8pts — what left the book was *signal*, not variance
(drawdown got slightly worse, −7.5% → −8.3%). The single-fold diagnostic
did not generalize: the tree used the 10d/63d horizons as trend context
for the physics block, not as duplicates. Kill; ledger 31 → 32. The
lesson is now symmetric: stacking dilutes (T5, T7) and amputating on a
single-fold diagnostic destroys — displacement must be earned with
full-history evidence. (This trial was run end-to-end by the
`quant-researcher` project agent as its validation run.)

## UPDATE (edge hunt, round 5): Guatemala supply, from the primary source
## — and cut

UN Comtrade paywalled its keyless tier, so the feed was rebuilt from the
origin: Banco de Guatemala's annual agricultural export-volume series
(`src/data/banguat.py`, pristine xlsx + committed CSV extraction,
publication-lagged to 01-Apr of the following year, 5 leakage tests).
The data is real and tells the right story — Guatemala's crop collapsed
−45% then −42% across 2024-25, the largest deficit in the 32-year series.

Two pre-registered trials (`scripts/gtm_trial.py`), ledger 32 → 34:

| trial | Sharpe | 90% CI | p(SR≤0) | AUC | hit vs base |
|---|---|---|---|---|---|
| T9 gtm-gbm | +0.49 | [−0.08, +1.04] | 0.08 | 0.540 | +2.5pts |
| T10 physics+gtm-gbm | +0.58 | [+0.07, +1.08] | 0.03 | 0.546 | +4.4pts |

**Cut, per the pre-registered rule.** T9's CI straddles zero, so the feed
does not pay solo; T10 is the fourth consecutive dilution result. Annual
frequency is the likely culprit — a once-a-year step function gives the
tree nothing to *time* that bid dispersion isn't already printing daily
at the auction. The loader, data and tests stay in the repo (the series
is the right cross-check for any future monthly wiring — SIECA/INE both
bot-wall their monthly volume data); the feature enters no configuration.

## Dataset v1.1 (15-Jul-2026): the lever, pulled — and a lesson in
## estimator variance

`scripts/refresh_spot.py` now operationalizes "more auction days": an
incremental crawl with an exact-overlap tripwire against the repaired
history (27/27 overlapping sessions matched to the paisa — no comma
corruption re-entered), append-only writes, and a full canonical rebuild
(ONI + rain preserved). It also survived a live site-markup change (the
archive's header row stopped being `<th>`; `_parse_html` now handles both
eras, regression-tested). Dataset: 3,148 → **3,155 auction days**,
through 15-Jul-2026.

Re-estimating the standing champion on v1.1 (same config — NOT a new
trial): **Sharpe +0.69, 90% CI [+0.19, +1.21], p(SR≤0) 1.5%, AUC 0.553,
DSR 0.51.** Seven added days can't move a Sharpe that much by themselves;
what moved was the walk-forward re-slicing (fold boundaries shift with n,
every GBM refits). Read it honestly: the +0.79 print carried fold-slicing
sensitivity the wide CI was already pricing in. The stable statement is
the one that survives re-slicing — *the physics book is positive with
~98% bootstrap confidence, in the +0.2…+1.2 range, and does not clear the
29-to-34-trial luck bar.* Point estimates are weather; the CI is climate.

## The slicing-robust estimate (16-Jul-2026): the honest headline drops
## to +0.43

`scripts/robust_estimate.py` replaced the single fold layout with a
family of six, declared ex ante (n_splits ∈ {5,6,7} × min_train offset
∈ {0,+63}), reporting the average — estimator refinement, not selection;
the DSR ledger is unchanged.

| layout | Sharpe |
|---|---|
| 5 folds, base | +0.30 |
| 5 folds, +63 | +0.01 |
| 6 folds, base (the one every prior table used) | +0.69 |
| 6 folds, +63 | +0.43 |
| 7 folds, base | +0.56 |
| 7 folds, +63 | +0.48 |

**Blended stream (per-day mean probability across layouts): Sharpe
+0.43, 90% CI [−0.01, +0.90], p(SR≤0) 5.5%, AUC 0.540, hit +4.2pts,
max drawdown −6.2%.**

Reported as found: a large part of the +0.79 print — and of every
single-layout number in this file — was fold-slicing luck. The layout
every prior table happened to use was the best of six. The claim that
survives is smaller and still real: *the physics book is positive with
~94% bootstrap confidence, classification skill is intact (+4.2pts,
AUC 0.54), drawdowns are shallow — and the 90% CI now grazes zero.*
Verdict: **a maybe, honestly priced.** This is why the prospective
forecast ledger (below) exists — backtest error bars have this many
degrees of freedom; a live track record has none.

## PROSPECTIVE VALIDATION (live since 16-Jul-2026)

`scripts/forecast.py` + `src/live/ledger.py`: every week the champion
emits P(up, 5d) for auction days whose outcomes do not yet exist, into
an append-only, hash-chained ledger (any edit or deletion of history
breaks every subsequent hash — `verify_chain` runs on every invocation
and in the test suite). Matured forecasts are scored into a second
chained ledger. `scripts/weekly_update.sh` (launchd agent) automates
crawl → verify → forecast → score. The running prospective scorecard
prints on every run; it is the number that will eventually settle this
project's question.

## UPDATE (edge hunt, round 6): the probability ensemble — a tie,
## called a tie

T12 (`scripts/ensemble_trial.py`, pre-registered, evaluated
slicing-robust from the start): per-day mean of the three calibrated
streams that cleared zero solo (physics, kalman, rain). Ledger 34 → 35.

Blended across the 6-layout family: **Sharpe +0.47, 90% CI [−0.03,
+1.01], AUC 0.544, hit +3.9pts, max_dd −7.1%** (single layouts −0.05…
+0.73). The pre-registered promote rule (beat +0.43) fires on the
letter — by +0.04, between two estimates whose intervals are half a
Sharpe wide. Reported as found: **a statistical tie.** Bagging did not
hurt (unlike all four stacking attempts) but bought no measurable edge
on this sample. Tiebreak goes to simplicity: the LIVE model stays
physics-gbm (one model, already forecasting on-chain); T12 stands in
the ledger as the nominal best number. DSR of T12 against all 35: 0.27.
The backtest modeling front is now CLOSED pending new information
(basis, forecast-rain) or live evidence.

## Forecast-rain groundwork (pilot passed)

`scripts/gefs_pilot.py` proved the keyless path to as-issued historical
rain forecasts: NOAA GEFS v12 reforecast on anonymous S3, per-variable
GRIB2, 27.5 MB/init, Idukki 5-day totals extract cleanly and pass the
seasonality sanity check (monsoon init 109 mm vs dry-season 0 mm).
Full reforecast-era backfill ≈ 36 GB streamed / ~1 h wall-clock, stored
as a small CSV; the 2020+ leg uses the operational archive with .idx
byte-range subsetting. The T13 forecast-rain trial is GO when scheduled.

## UPDATE (edge hunt, round 7): MCX basis — the coverage problem, faced
## honestly — cut

The MCX Bhavcopy backfill landed: `data/raw/mcx/CommodityWise_CARDAMOM_
backfill.csv` (1,305 contract-day rows since the 2025-07-29 relaunch),
parsed by the existing `MCXBhavcopyLoader` (unchanged) and wired into
`market.parquet` via `scripts/merge_mcx.py` (same pattern as
`merge_rain.py` — touches only the futures/basis columns, spot/ONI/rain
untouched). `basis_pct`/`basis_chg` were already declared in
`CORE_FEATURE_COLS` and already causally shifted — dark, not absent,
since the very first version of `src/features/engineering.py`.

**The coverage problem, stated before any model ran:** basis_pct is
non-null for 232 of 3,153 feature rows (7.4%) — 2025-07-30 → 2026-07-11 —
inside an 11-year dataset. It is NaN across the entire pre-relaunch
history and the whole 2021–2025 suspension gap, by construction. Checked
analytically first: under the project's standard 6-fold vehicle, every
fold's training window ends before row 2,873 (the first non-null row) —
zero folds ever see a non-missing basis value while fitting. This isn't
hypothetical: running the naive full-history champion+basis config
(`scripts/mcx_basis_trial.py`'s diagnostic, reusing
`scripts/ensemble_trial.py`'s own physics member matrix unmodified)
doesn't produce a misleading Sharpe — it **throws inside sklearn's
histogram binning** (`ValueError: window shape cannot be larger than
input array shape`), because a shallow GBM's binner needs ≥2 distinct
non-missing training values per feature and gets zero. A full-history
Sharpe for this feature was never going to be an honest number; it isn't
even a computable one.

**T14 mcx-basis-gbm** (`scripts/mcx_basis_trial.py`, pre-registered,
option-(a)-literal): the entire train+calibrate+test universe restricted
to the 281 covered-window rows (2025-07-29 onward; momentum/vol/physics
features still use their full-history rolling windows, only the CV row
*subset* is restricted). `PurgedWalkForward(n_splits=3, min_train=140,
purge=5, embargo=5)` — min_train derived from "leaves ≥75 rows after the
pipeline's own calibration carve-out," not swept — three 47-day test
blocks, 141 OOS days total (2026-01-20 → 2026-07-13). Same isotonic
calibration, tranched 5d rebalancing, 15bps costs as every other trial.
Champion-without-basis evaluated on the *identical* folds as the paired
control (not separately counted — same precedent as
`robust_estimate.py`'s re-slicing: "estimator refinement, not
selection"). Ledger 35 → 36.

| config | Sharpe | 90% CI | p(SR≤0) | AUC | hit vs base | n OOS |
|---|---|---|---|---|---|---|
| control (champion, no basis, same folds) | −1.24 | [−3.21, +0.94] | 0.850 | 0.466 | −2.1pts | 141 |
| **T14 mcx-basis-gbm** | **−0.59** | [−2.17, +1.54] | 0.705 | 0.521 | −0.7pts | 141 |

DSR of T14 against the 36-trial ledger: **0.157** (expected max Sharpe
from luck alone across 36 zero-skill trials: 0.75).

**Cut.** Read honestly: basis is directionally *consistent* with the
hypothesis on this sample — it beats the no-basis control on the
identical folds (+0.65 Sharpe, AUC 0.466 → 0.521, hit-vs-base −2.1 →
−0.7pts) — but the absolute number is negative, the 90% CI spans over 3.5
Sharpe points and buries zero deep inside it, and 141 OOS days (~28
non-overlapping 5-day bets) is nowhere near enough to distinguish this
from noise. This is not "no edge after costs" (T3/T4's mechanism); it is
"no sample yet" — a different, smaller kind of honest negative. The
champion is unchanged: **physics-gbm / T12 ensemble stand as reported in
round 6.** Revisit when the covered window is materially longer (each
additional trading year roughly doubles it) or drop it for good if it
still doesn't clear zero with a real sample.

## Roadmap implied by the numbers

~~Horizon-matched + calibration~~ ~~auction physics~~ ~~rain~~
~~displacement~~ ~~Guatemala~~ ~~refresh + live ledger~~ ~~slicing-
robust estimator~~ ~~probability ensemble~~ ~~MCX basis (cut, too
little history since relaunch)~~ (all done, all counted). Open: T13
forecast-rain backfill + trial (pilot passed, ~1 h of streaming), the
live ledgers accruing ~250 scored forecasts/year, and MCX basis itself —
revisit, don't re-litigate, once the covered window is longer. Ledger
stands at 36 trials.

*Machine-verifiable provenance: every number above regenerates from
`data/processed/market.parquet` via `python run.py`, `scripts/analyze.py`,
`scripts/horizon_experiment.py` and `scripts/edge_hunt.py`.*
