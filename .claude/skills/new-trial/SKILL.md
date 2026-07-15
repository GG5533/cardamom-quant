---
name: new-trial
description: The only allowed way to evaluate a new model, feature block, or trading config in cardamom-quant — pre-register, leakage-test, run in the standard vehicle, grow the DSR ledger.
---

# Adding a trial without lying to yourself

This repo's brand is that its numbers survive hostile review. Every trial
follows the same five steps — no exceptions for "quick looks", because a
quick look you liked IS a trial.

## 1. Pre-register before running

Write the hypothesis and its mechanism into the experiment script's
docstring (see `scripts/edge_hunt.py` for the pattern) BEFORE the first
run. One config = one trial. A swept parameter is one trial per point —
prefer deriving parameters from theory (the OU band is derived from
first-passage calculus, not swept) so the count stays small. If you add a
trial after seeing results (sequencing), say so in the docstring — T5 in
edge_hunt.py is the disclosed example.

## 2. Leakage test first, feature second

Every feature gets a "mutate day-t inputs, assert day-t feature unchanged"
test before it touches a model. Pattern: `test_edge_hunt.py::
test_physics_features_are_strictly_causal`. Expanding norms must use
strictly-past data only (see `inventory_overhang` + its past-years-only
test). Publication-lagged feeds (ONI 2m, Comtrade 3m) keep their lags.
Fold-fitted transforms (Kalman, calibration) fit on `[:train_end]` only
and run causally over the full series — filter, never smoother.

## 3. Evaluate in the standard vehicle

Tranched 5-day rebalance, per-fold isotonic-calibrated GBM probabilities,
15bps costs, `PurgedWalkForward(n_splits=6, min_train=max(400, n//4))`.
Reuse `calibrated_fold_proba` + `evaluate_stream` in `scripts/edge_hunt.py`.
Never report a single-anchor 5d backtest — anchor luck is worth ±0.4
Sharpe on this data (measured).

## 4. Grow the ledger — even for failures

`PRIOR_LEDGER` in `scripts/edge_hunt.py` and the ledger line in
RESULTS_REAL.md hold every Sharpe ever evaluated (29 after round 2).
Append yours, rerun `deflated_sharpe`, and report DSR against the full
count. A failed trial still counts — deleting it would flatter every
future DSR.

## 5. Report as found

Add the trial to RESULTS_REAL.md with Sharpe, 90% block-bootstrap CI,
p(SR≤0), and DSR. Failures get written up with the same care as wins
(see T3/T4 in the round-2 section: killed, with the mechanism of death
stated). Update the one-line verdict only if the DSR says so:
below ~0.9 the honest label stays "maybe".
