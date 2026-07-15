# HANDOFF — cardamom-quant → VS Code

**Date:** July 7, 2026. Everything below is current as of this session's end.

> **STATUS UPDATE (local VS Code session):** first-30-minutes done (py3.12
> venv, 49→56 tests green, ONI ported into root `run.py`, stale sample CSVs
> deleted, git initialized). Priority #1 + #2 are DONE — see
> `scripts/horizon_experiment.py` and the 07-Jul update section in
> `RESULTS_REAL.md`: horizon-matched 5d rebalancing + isotonic calibration,
> anchor-free tranched book Sharpe +0.57, 90% CI [+0.05, +1.12], DSR 0.49
> over the now-24-trial ledger — "a defensible maybe." LINKEDIN_POST.md
> numbers updated to match. Remaining: priority #3 (wire feeds one at a
> time) and #4 (ship).
>
> **ROUND 2 (edge hunt):** `scripts/edge_hunt.py` — 5 more counted trials.
> New headline: **physics-gbm** (bid dispersion + inventory overhang +
> Hurst, calibrated, tranched 5d) **Sharpe +0.79, 90% CI [+0.28, +1.29],
> DSR 0.63 over 29 trials** — "strengthening maybe." Kalman anomaly helps
> alone (+0.67) but not on top; OU bands and conformal gate killed and
> ledgered. New machinery: `src/features/auction_physics.py`,
> `src/models/kalman_seasonal.py`, `src/models/ou_bands.py`,
> `run_weights_backtest`, 8 new tests (64 total). Project skills/agents in
> `.claude/` (verify, new-trial, quant-researcher).
>
> **ROUND 3 (rain):** IMD gridded rain 2010–2025 wired into market.parquet
> via `scripts/merge_rain.py` (do NOT use `build_dataset.py --refresh` — it
> would rebuild spot from the unrepaired loader cache). Kerala-flood
> acceptance check passed. T6 rain-gbm +0.64 (AUC 0.572, +6.3pts — best
> classification of the project) pays solo; champion unchanged:
> **physics-gbm +0.79, DSR 0.63 vs 31-trial ledger**. Third dilution
> result — future features must displace, not stack.
>
> **ROUND 4 (displacement, agent-run):** T8 dropped mom_10/mom_63 from the
> champion — KILLED (+0.63, signal left not variance; single-fold dAUC
> diagnostics don't generalize). Ledger 32. Run end-to-end by the
> `.claude/agents/quant-researcher` agent — the trial-discipline
> automation works.

## What this is

AI price-direction model for Indian small cardamom, built as a quant-AI
portfolio piece. Trains on 12 years of real Spices Board e-auction spot data
(the MCX future was suspended 2021→relaunched 2025-07-29, so it has only ~1y
of history and serves as basis/tradability overlay, not training data).
Full narrative: `README.md` → `STEP1_RESEARCH_AND_PLAN.md` →
`ALT_DATA_RESEARCH.md` → `RESULTS_REAL.md` (read in that order).

## State: what is DONE

- **Real dataset in repo:** `data/processed/market.parquet` — 3,148 auction
  days, 2014-11-07 → 2026-07-07, from 5,655 repaired sessions (crawled live
  07-Jul-2026; 15 bad source rows quarantined in
  `data/raw/browser/sessions_repaired.csv.rejected`). Raw sessions backup
  also in `~/Downloads/spices_sessions_full.csv`. Real NOAA ONI 2010→Feb-2026
  in `data/raw/climate/oni.ascii.txt`.
- **Pipeline:** loaders (`src/data/`), core+alt features (`src/features/`),
  purged walk-forward (`src/validation/`), 3 models (`src/models/`),
  vol-targeted backtest w/ 15bps costs (`src/backtest/`), robustness/
  calibration/SHAP/capacity (`src/analysis/`), dashboard (`app.py`),
  49 offline tests (`tests/`), `run.py`, `scripts/analyze.py`.
- **Real results (see RESULTS_REAL.md for the full table):** GBM has real
  skill (+4.2pts vs base, AUC 0.553 OOS) but nothing beats the seasonal
  baseline after costs (best Sharpe +0.30, all bootstrap CIs straddle 0,
  best DSR 0.38). Alt features HURT on real data. Honest verdict: signal
  exists, daily-rebalance turnover eats it.

## First 30 minutes in VS Code

```bash
cd cardamom-quant
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-data.txt scikit-learn scipy shap streamlit
python -m pytest tests/ -q     # expect 49 passed
python run.py                  # picks up data/processed/market.parquet -> [REAL] scorecard
python scripts/analyze.py      # ablation / bootstrap / DSR / calibration / regimes / SHAP
streamlit run app.py           # dashboard, green REAL banner
git init && git add -A && git commit -m "cardamom-quant v1: real-data scorecard"
```

Note: `run.py` in the repo root is the canonical one.
`scripts/run_real_reference.py` is the exact copy that produced
RESULTS_REAL.md (includes the ONI wiring patch — diff them and port the
ONI-loading block into root `run.py`, it's ~7 lines).

## Priority queue (in order of expected value)

1. **Horizon-matched rebalancing + conviction threshold** — THE experiment.
   The 5d-label signal is real; daily rebalancing burns it. Add to
   `src/backtest/engine.py`: rebalance every `HORIZON` days (or when
   |p−0.5| > threshold, e.g. 0.10), hold otherwise. Sweep threshold ∈
   {0, .05, .10, .15} × rebalance ∈ {1d, 5d}. Report the 2×4 grid with
   bootstrap CIs. If any cell's CI clears zero, that's the headline.
2. **Calibrate GBM probabilities** (sklearn `CalibratedClassifierCV`,
   isotonic, cv="prefit" per fold) — Brier was 0.270 vs 0.250 climatology;
   sizing on uncalibrated p is self-harm. Do before/with #1.
3. **Wire remaining feeds one at a time through `scripts/analyze.py`'s
   ablation** (each must pay OOS or gets cut):
   - IMD rain: `python scripts/build_dataset.py --refresh` (imdlib download
     works from your machine; it was network-blocked in the cloud sandbox).
   - MCX basis: manually download daily Bhavcopy CSVs since 2025-07-29 from
     mcxindia.com/market-data/bhavcopy into `data/raw/mcx/` — parser handles
     both file eras, builds roll-safe continuous series.
   - Guatemala/FX: fetchers ready in `src/data/comtrade.py` / `fx.py`
     (Comtrade free key via env `COMTRADE_KEY` recommended).
4. **Ship:** GitHub push, dashboard screenshot, flagship chart (12y spot
   history, annotate 2019 flood spike + 2021-25 futures suspension window),
   then `LINKEDIN_POST.md` (numbers already filled in — update if #1
   changes the verdict).

## Gotchas / tribal knowledge

- **indianspices.com numbers contain thousands-separator commas** (mixed
  3-digit and Indian lakh grouping). Naive parsers silently corrupt ~22% of
  rows. `scripts/repair_sessions.py` fixes via unique-sane-partition; the
  15 rejects are lakh-grouped rows + one source typo (`3666..00`). If you
  re-crawl with `SpicesBoardLoader.fetch()` (pandas `read_html`), verify
  against `data/processed/spot_daily_full.csv`.
- `scripts/ingest_browser_dump.py` rebuilds `market.parquet` from a sessions
  CSV — use after any re-crawl.
- **Leakage discipline is enforced by tests** — if you add a feature, add
  the corresponding "mutate today, assert today unchanged" test
  (`tests/test_signal_layer.py` has the pattern). ONI/Comtrade features are
  publication-lagged (2m/3m); don't "fix" that.
- The 6-trial deflated-Sharpe count in `RESULTS_REAL.md` must grow with
  every new model/config you evaluate. Keep the ledger honest.
- `requirements-data.txt` / `requirements-analysis.txt` got file-locked
  during a cloud session — verify they contain: data deps + scikit-learn,
  scipy, shap, streamlit; fix by hand if truncated.
- `data/processed/spot_daily_real.csv` and `spices_sessions_real.csv` are a
  stale 11-day sample from 02-Jul; the full versions are
  `spot_daily_full.csv` / `sessions_full_repaired.csv`. Delete the stale
  ones on first commit.
- MCX cardamom contract: 100kg lot, ₹/kg ex-Vandanmedu, compulsory
  delivery, DPL 4%+2%. Regime column must never let features cross the
  2021–2025 gap (validated in `MCXBhavcopyLoader.validate`).

## One-line status if anyone asks

Pipeline desk-grade and tested; 12 years of real spot data in repo; verdict
so far "real signal, no edge after costs at daily rebalancing"; next
experiment (horizon-matched trading) is specced above and is the difference
between ending on a no and ending on a defensible maybe.
