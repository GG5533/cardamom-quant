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

## Roadmap implied by the numbers

~~Horizon-matched rebalancing + conviction thresholds + probability
calibration~~ (done, above), then the unwired feeds (rain, basis once MCX
files are dropped, Guatemala) evaluated one at a time against the ablation
harness — each addition must pay for itself out-of-sample or it goes. Every
new configuration grows the 24-trial DSR ledger.

*Machine-verifiable provenance: every number above regenerates from
`data/processed/market.parquet` via `python run.py`, `scripts/analyze.py`
and `scripts/horizon_experiment.py`.*
