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

## Roadmap implied by the numbers

Horizon-matched rebalancing + conviction thresholds + probability
calibration, then the unwired feeds (rain, basis once MCX files are dropped,
Guatemala) evaluated one at a time against the ablation harness — each
addition must pay for itself out-of-sample or it goes.

*Machine-verifiable provenance: every number above regenerates from
`data/processed/market.parquet` via `python run.py` and the analysis
snippet in git history.*
