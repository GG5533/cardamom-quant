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

## Roadmap implied by the numbers

~~Horizon-matched rebalancing + calibration~~ (done). ~~Auction physics~~
(done — champion). ~~Rain~~ (done — pays as signal, kept). Next: MCX basis
(manual Bhavcopy drop into `data/raw/mcx/`), Guatemala/FX via Comtrade
key, and feature *selection* rather than feature *stacking* — three
dilution results say the marginal feature must now displace a weaker one,
not join it. Every new configuration grows the 31-trial DSR ledger.

*Machine-verifiable provenance: every number above regenerates from
`data/processed/market.parquet` via `python run.py`, `scripts/analyze.py`,
`scripts/horizon_experiment.py` and `scripts/edge_hunt.py`.*
