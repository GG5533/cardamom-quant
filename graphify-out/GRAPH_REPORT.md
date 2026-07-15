# Graph Report - cardamom-quant  (2026-07-15)

## Corpus Check
- 63 files · ~33,642 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 462 nodes · 674 edges · 36 communities (28 shown, 8 thin omitted)
- Extraction: 74% EXTRACTED · 26% INFERRED · 0% AMBIGUOUS · INFERRED: 176 edges (avg confidence: 0.79)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `338e031e`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 17|Community 17]]
- [[_COMMUNITY_Community 18|Community 18]]
- [[_COMMUNITY_Community 24|Community 24]]
- [[_COMMUNITY_Community 25|Community 25]]
- [[_COMMUNITY_Community 26|Community 26]]
- [[_COMMUNITY_Community 27|Community 27]]
- [[_COMMUNITY_Community 28|Community 28]]
- [[_COMMUNITY_Community 29|Community 29]]
- [[_COMMUNITY_Community 31|Community 31]]
- [[_COMMUNITY_Community 32|Community 32]]
- [[_COMMUNITY_Community 33|Community 33]]
- [[_COMMUNITY_Community 36|Community 36]]

## God Nodes (most connected - your core abstractions)
1. `BacktestConfig` - 17 edges
2. `run_backtest()` - 17 edges
3. `main()` - 16 edges
4. `SpicesBoardLoader` - 15 edges
5. `MCXBhavcopyLoader` - 13 edges
6. `main()` - 12 edges
7. `build_features()` - 12 edges
8. `IMDRainfallLoader` - 12 edges
9. `ValidationError` - 12 edges
10. `PurgedWalkForward` - 12 edges

## Surprising Connections (you probably didn't know these)
- `main()` --calls--> `parse_oni()`  [INFERRED]
  run.py → src/data/climate_indices.py
- `main()` --calls--> `build_alt_features()`  [INFERRED]
  run.py → src/features/alt_features.py
- `main()` --calls--> `run_backtest()`  [INFERRED]
  run.py → src/backtest/engine.py
- `main()` --calls--> `BacktestConfig`  [INFERRED]
  run.py → src/backtest/engine.py
- `run_cv()` --calls--> `build_alt_features()`  [INFERRED]
  app.py → src/features/alt_features.py

## Communities (36 total, 8 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.06
Nodes (46): deflated_sharpe(), DSR: PSR against the expected-max-Sharpe of everything we tried.      all_trial_, load_market(), Cardamom Quant — interactive dashboard.      pip install streamlit     streamlit, run_cv(), load_dataset(), main(), cardamom-quant — end-to-end run.      python run.py                 # real data (+38 more)

### Community 1 - "Community 1"
Cohesion: 0.07
Nodes (37): build_physics_features(), _crop_year(), _day_in_crop_year(), hurst_rs(), inventory_overhang(), Auction-physics features — signal mined from columns the pipeline never touched,, Assemble the block on the market calendar; all columns lag-safe., R/S Hurst exponent estimate (three-scale log-log slope). (+29 more)

### Community 2 - "Community 2"
Cohesion: 0.07
Nodes (36): brier_score(), calibration_summary(), calibration_table(), enso_phase(), isotonic_calibrator(), Probability calibration + regime-conditional performance.  Calibration: a 0.65 f, Fit an isotonic map raw p -> calibrated p on a held-out slice.      The slice mu, elnino' (>= +0.5), 'lanina' (<= -0.5), else 'neutral'. (+28 more)

### Community 3 - "Community 3"
Cohesion: 0.09
Nodes (25): add_calendar_features(), _days_to_next(), Demand-calendar features — the moving seasonals that day-of-year misses.  The ou, Signed day count to the next occurrence of an annual event.      Positive = even, Feature block for a given trading calendar. All ex-ante knowable., fetch_dmi(), fetch_oni(), parse_oni() (+17 more)

### Community 4 - "Community 4"
Cohesion: 0.05
Nodes (37): ABC, BaseLoader, BaseLoader, fetch(), parse(), BaseLoader: the common contract every real-data feed implements.  Design princip, Raised when a loader's output violates its schema contract., Fetch → parse → validate → load, with an immutable raw cache. (+29 more)

### Community 5 - "Community 5"
Cohesion: 0.08
Nodes (25): build_market_dataset(), load_futures(), load_rain(), load_spot(), Real-data facade — the swap-in point that replaces synthetic.py.  Public API (wh, Join the three feeds on a master calendar, compute basis honestly., aggregate_daily(), _normalise() (+17 more)

### Community 6 - "Community 6"
Cohesion: 0.11
Nodes (20): fetch_usdinr(), parse_fred_csv(), FX competitiveness — USD/INR from FRED (no API key needed).  Indian cardamom com, fredgraph.csv -> daily Series; '.' means missing., FX features on the trading calendar (ffill weekends/holidays, max 5d)., to_daily_features(), auction_microstructure(), build_alt_features() (+12 more)

### Community 7 - "Community 7"
Cohesion: 0.14
Nodes (11): mean_abs_shap(), SHAP interpretability — per-prediction attribution on top of the permutation imp, SHAP values for a fitted model from src/models/baselines.py.      logistic  -> L, Global importance: mean |SHAP| per feature, descending., shap_report(), shap_values_for(), make_gbm(), Models. The rule of this repo: ML must beat the dumb seasonal rule or we say so (+3 more)

### Community 8 - "Community 8"
Cohesion: 0.14
Nodes (13): 1. The critical finding (this reshapes the project — for the better), 2.1 Spices Board e-auction (spot) — PRIMARY, verified working, 2.2 MCX futures (Bhavcopy) — the loader Step 1 names, 2.3 Rainfall (IMD) — the weather signal, 2.4 Rejected alternatives (document these — rejection rationale is showcase material), 2. Data source research (verified where possible), 3. `loaders.py` design, 4. Build sequence (with acceptance criteria) (+5 more)

### Community 9 - "Community 9"
Cohesion: 0.21
Nodes (12): capacity_report(), market_capacity(), participation_table(), Capacity analysis — could this strategy actually be run, and at what size?  A Sh, Recent tradable-size envelope from real market data., For each assumed capital: turnover-driven daily participation in the     auction, Tests for the capacity module (offline)., test_capacity_report_renders() (+4 more)

### Community 10 - "Community 10"
Cohesion: 0.22
Nodes (8): code:bash (cd cardamom-quant), First 30 minutes in VS Code, Gotchas / tribal knowledge, HANDOFF — cardamom-quant → VS Code, One-line status if anyone asks, Priority queue (in order of expected value), State: what is DONE, What this is

### Community 11 - "Community 11"
Cohesion: 0.25
Nodes (7): 1. Moving demand calendars — `src/data/calendars.py`, 2. Climate teleconnections — `src/data/climate_indices.py`, Cardamom Quant — Alternative-Data Research (beyond the obvious), Implemented round 2 (code + 9 more tests), Implemented today (code + 8 new tests, all offline), Remaining roadmap, Why this layer wins interviews

### Community 12 - "Community 12"
Cohesion: 0.25
Nodes (7): Build everything, code:bash (pip install -r requirements-data.txt), Known limitations (also candidates for the LinkedIn write-up), MCX files: what to drop where, Real-data ingestion layer, The three loaders, Why the spot market is the backbone

### Community 13 - "Community 13"
Cohesion: 0.29
Nodes (6): code:bash (cd cardamom-quant), Data status — what's real, what's pending, and why, Real data in the repo now (`data/`), UPDATE 07-Jul-2026: full backfill COMPLETE (via browser crawl), Verified behaviors from today's real-data run, Why the rest isn't here yet — and the one command that finishes it

### Community 14 - "Community 14"
Cohesion: 0.33
Nodes (5): Attachments (in order of impact), LinkedIn post — Cardamom Quant, Main post, Posting notes, Short variant (for comments / reposts)

### Community 15 - "Community 15"
Cohesion: 0.33
Nodes (5): Cardamom Quant, Honesty guardrails, Layout, Run, The data story (the interesting part)

### Community 16 - "Community 16"
Cohesion: 0.53
Nodes (5): join_ok(), main(), partitions(), Repair thousands-separator commas in the browser-crawled sessions CSV.  Each dat, sane()

### Community 17 - "Community 17"
Cohesion: 0.2
Nodes (9): REAL out-of-sample results — July 7, 2026, Roadmap implied by the numbers, signal, not as the champion, The findings — reported as found, The scorecard (purged walk-forward, 6 folds, after 15bps costs), UPDATE 07-Jul-2026 (local): horizon-matched trading — the "maybe", UPDATE (edge hunt, round 2): auction physics is the new headline, UPDATE (edge hunt, round 3): the rain feed is wired — and it pays as (+1 more)

### Community 31 - "Community 31"
Cohesion: 0.17
Nodes (21): BacktestConfig, Backtest engine: conviction sizing, vol targeting, leverage cap, costs.  Mechani, proba_up: model P(up) indexed by date; daily_returns: same calendar., proba_up: model P(up) indexed by date; daily_returns: same calendar., Same execution/cost mechanics for a strategy that emits target weights     direc, run_backtest(), run_weights_backtest(), test_run_weights_backtest_mechanics() (+13 more)

### Community 32 - "Community 32"
Cohesion: 0.18
Nodes (10): 1. Test suite (fast, always first), 2. Scorecard reproduction (the numbers are the product), 3. Experiment scripts regenerate their published tables, 4. Dashboard boots with the REAL banner, 5. Honesty checklist (read RESULTS_REAL.md ledger line), code:bash (.venv/bin/python -m pytest tests/ -q), code:bash (.venv/bin/python run.py            # [REAL] tag, not [SYNTHE), code:bash (.venv/bin/python scripts/analyze.py             # ablation/D) (+2 more)

### Community 33 - "Community 33"
Cohesion: 0.29
Nodes (6): 1. Pre-register before running, 2. Leakage test first, feature second, 3. Evaluate in the standard vehicle, 4. Grow the ledger — even for failures, 5. Report as found, Adding a trial without lying to yourself

## Knowledge Gaps
- **171 isolated node(s):** `cardamom-quant — end-to-end run.      python run.py                 # real data`, `Cardamom Quant — interactive dashboard.      pip install streamlit     streamlit`, `Leakage + sanity tests for the edge-hunt round: auction physics, Kalman seasonal`, `Mutating TODAY's auction must not change TODAY's features.`, `Tests for the cross-market/macro/microstructure signal layer (offline).` (+166 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **8 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `parse_oni()` connect `Community 3` to `Community 0`?**
  _High betweenness centrality (0.223) - this node is a cross-community bridge._
- **Why does `SpicesBoardLoader` connect `Community 5` to `Community 3`, `Community 4`?**
  _High betweenness centrality (0.207) - this node is a cross-community bridge._
- **Why does `main()` connect `Community 3` to `Community 5`?**
  _High betweenness centrality (0.199) - this node is a cross-community bridge._
- **Are the 16 inferred relationships involving `BacktestConfig` (e.g. with `main()` and `run_cv()`) actually correct?**
  _`BacktestConfig` has 16 INFERRED edges - model-reasoned connections that need verification._
- **Are the 14 inferred relationships involving `run_backtest()` (e.g. with `main()` and `run_cv()`) actually correct?**
  _`run_backtest()` has 14 INFERRED edges - model-reasoned connections that need verification._
- **Are the 13 inferred relationships involving `main()` (e.g. with `build_features()` and `forward_returns()`) actually correct?**
  _`main()` has 13 INFERRED edges - model-reasoned connections that need verification._
- **Are the 6 inferred relationships involving `SpicesBoardLoader` (e.g. with `BaseLoader` and `ValidationError`) actually correct?**
  _`SpicesBoardLoader` has 6 INFERRED edges - model-reasoned connections that need verification._