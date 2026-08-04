# Graph Report - agent-a673b3a56aa887d15  (2026-08-04)

## Corpus Check
- 80 files · ~170,219 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 689 nodes · 1196 edges · 55 communities (45 shown, 10 thin omitted)
- Extraction: 81% EXTRACTED · 19% INFERRED · 0% AMBIGUOUS · INFERRED: 226 edges (avg confidence: 0.79)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `a2d9a5c6`
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
- [[_COMMUNITY_Community 35|Community 35]]
- [[_COMMUNITY_Community 36|Community 36]]
- [[_COMMUNITY_Community 37|Community 37]]
- [[_COMMUNITY_Community 38|Community 38]]
- [[_COMMUNITY_Community 39|Community 39]]
- [[_COMMUNITY_Community 43|Community 43]]
- [[_COMMUNITY_Community 44|Community 44]]
- [[_COMMUNITY_Community 52|Community 52]]
- [[_COMMUNITY_Community 53|Community 53]]

## God Nodes (most connected - your core abstractions)
1. `REAL out-of-sample results — July 7, 2026` - 21 edges
2. `build_features()` - 19 edges
3. `REAL out-of-sample results — July 7, 2026` - 19 edges
4. `BacktestConfig` - 18 edges
5. `run_backtest()` - 18 edges
6. `main()` - 17 edges
7. `SpicesBoardLoader` - 17 edges
8. `PurgedWalkForward` - 17 edges
9. `deflated_sharpe()` - 14 edges
10. `build_physics_features()` - 14 edges

## Surprising Connections (you probably didn't know these)
- `test_parse_comtrade_sums_hs_codes()` --calls--> `parse_comtrade()`  [INFERRED]
  tests/test_signal_layer.py → /Users/samihabbal/Documents/Claude/Projects/My personal Profile and who i am/cardamom-quant/src/data/comtrade.py
- `main()` --calls--> `build_forecast_rain_features()`  [INFERRED]
  scripts/forecast_rain_trial.py → src/features/forecast_rain.py
- `test_parse_fred_csv_missing_dot()` --calls--> `parse_fred_csv()`  [INFERRED]
  tests/test_signal_layer.py → /Users/samihabbal/Documents/Claude/Projects/My personal Profile and who i am/cardamom-quant/src/data/fx.py
- `test_fx_features_ffill_limited()` --calls--> `parse_fred_csv()`  [INFERRED]
  tests/test_signal_layer.py → /Users/samihabbal/Documents/Claude/Projects/My personal Profile and who i am/cardamom-quant/src/data/fx.py
- `test_microstructure_is_strictly_causal()` --calls--> `auction_microstructure()`  [INFERRED]
  tests/test_signal_layer.py → /Users/samihabbal/Documents/Claude/Projects/My personal Profile and who i am/cardamom-quant/src/features/alt_features.py

## Communities (55 total, 10 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.05
Nodes (58): ablation_table(), block_bootstrap_sharpe(), deflated_sharpe(), expected_max_sharpe_annual(), probabilistic_sharpe(), Robustness statistics — is the Sharpe distinguishable from luck?  Three tools mo, results: {variant_name: metrics_dict} -> tidy comparison frame., Circular block bootstrap of the annualized Sharpe ratio. (+50 more)

### Community 1 - "Community 1"
Cohesion: 0.11
Nodes (30): filter_anomaly(), fit_params(), _fourier(), half_life_days(), kalman_features(), KalmanParams, kappa(), Structural decomposition of log spot: level + seasonal + AR(1) anomaly.  Ported/ (+22 more)

### Community 2 - "Community 2"
Cohesion: 0.14
Nodes (25): brier_score(), calibration_summary(), calibration_table(), enso_phase(), isotonic_calibrator(), Probability calibration + regime-conditional performance.  Calibration: a 0.65 f, Fit an isotonic map raw p -> calibrated p on a held-out slice.      The slice mu, elnino' (>= +0.5), 'lanina' (<= -0.5), else 'neutral'. (+17 more)

### Community 3 - "Community 3"
Cohesion: 0.1
Nodes (26): add_calendar_features(), _days_to_next(), Demand-calendar features — the moving seasonals that day-of-year misses.  The ou, Signed day count to the next occurrence of an annual event.      Positive = even, Feature block for a given trading calendar. All ex-ante knowable., fetch_dmi(), fetch_oni(), parse_oni() (+18 more)

### Community 4 - "Community 4"
Cohesion: 0.05
Nodes (41): ABC, BaseLoader, BaseLoader, fetch(), parse(), BaseLoader: the common contract every real-data feed implements.  Design princip, Raised when a loader's output violates its schema contract., Fetch → parse → validate → load, with an immutable raw cache. (+33 more)

### Community 5 - "Community 5"
Cohesion: 0.11
Nodes (24): aggregate_daily(), _normalise(), _parse_html(), SpicesBoardLoader — small-cardamom e-auction archive (the spot backbone).  The S, Session-level table across all cached pages, deduped., Daily aggregate: quantity-weighted average price + supply columns., Session-level table across all cached pages, deduped., Daily aggregate: quantity-weighted average price + supply columns. (+16 more)

### Community 6 - "Community 6"
Cohesion: 0.08
Nodes (39): fetch_usdinr(), parse_fred_csv(), FX competitiveness — USD/INR from FRED (no API key needed).  Indian cardamom com, fredgraph.csv -> daily Series; '.' means missing., FX features on the trading calendar (ffill weekends/holidays, max 5d)., to_daily_features(), auction_microstructure(), build_alt_features() (+31 more)

### Community 7 - "Community 7"
Cohesion: 0.12
Nodes (23): _canon(), _chain_hash(), ChainedCsv, forecast_ledger(), outcome_ledger(), Tamper-evident forecast ledger — the prospective-validation backbone.  A backtes, Append-only CSV where every row extends a hash chain., Recompute every hash; raise on the first broken link. (+15 more)

### Community 8 - "Community 8"
Cohesion: 0.13
Nodes (13): 1. The critical finding (this reshapes the project — for the better), 2.1 Spices Board e-auction (spot) — PRIMARY, verified working, 2.2 MCX futures (Bhavcopy) — the loader Step 1 names, 2.3 Rainfall (IMD) — the weather signal, 2.4 Rejected alternatives (document these — rejection rationale is showcase material), 2. Data source research (verified where possible), 3. `loaders.py` design, 4. Build sequence (with acceptance criteria) (+5 more)

### Community 9 - "Community 9"
Cohesion: 0.24
Nodes (12): capacity_report(), market_capacity(), participation_table(), Capacity analysis — could this strategy actually be run, and at what size?  A Sh, Recent tradable-size envelope from real market data., For each assumed capital: turnover-driven daily participation in the     auction, Tests for the capacity module (offline)., test_capacity_report_renders() (+4 more)

### Community 10 - "Community 10"
Cohesion: 0.2
Nodes (8): code:bash (cd cardamom-quant), First 30 minutes in VS Code, Gotchas / tribal knowledge, HANDOFF — cardamom-quant → VS Code, One-line status if anyone asks, Priority queue (in order of expected value), State: what is DONE, What this is

### Community 11 - "Community 11"
Cohesion: 0.22
Nodes (7): 1. Moving demand calendars — `src/data/calendars.py`, 2. Climate teleconnections — `src/data/climate_indices.py`, Cardamom Quant — Alternative-Data Research (beyond the obvious), Implemented round 2 (code + 9 more tests), Implemented today (code + 8 new tests, all offline), Remaining roadmap, Why this layer wins interviews

### Community 12 - "Community 12"
Cohesion: 0.22
Nodes (7): Build everything, code:bash (pip install -r requirements-data.txt), Known limitations (also candidates for the LinkedIn write-up), MCX files: what to drop where, Real-data ingestion layer, The three loaders, Why the spot market is the backbone

### Community 13 - "Community 13"
Cohesion: 0.25
Nodes (6): code:bash (cd cardamom-quant), Data status — what's real, what's pending, and why, Real data in the repo now (`data/`), UPDATE 07-Jul-2026: full backfill COMPLETE (via browser crawl), Verified behaviors from today's real-data run, Why the rest isn't here yet — and the one command that finishes it

### Community 14 - "Community 14"
Cohesion: 0.29
Nodes (5): Attachments (in order of impact), LinkedIn post — Cardamom Quant, Main post, Posting notes, Short variant (for comments / reposts)

### Community 15 - "Community 15"
Cohesion: 0.29
Nodes (5): Cardamom Quant, Honesty guardrails, Layout, Run, The data story (the interesting part)

### Community 16 - "Community 16"
Cohesion: 0.62
Nodes (5): join_ok(), main(), partitions(), Repair thousands-separator commas in the browser-crawled sessions CSV.  Each dat, sane()

### Community 17 - "Community 17"
Cohesion: 0.05
Nodes (42): — and cut, and cut, called a tie, Dataset v1.1 (15-Jul-2026): the lever, pulled — and a lesson in, estimator variance, Forecast-rain groundwork (pilot passed), PROSPECTIVE VALIDATION (live since 16-Jul-2026), REAL out-of-sample results — July 7, 2026 (+34 more)

### Community 31 - "Community 31"
Cohesion: 0.1
Nodes (31): BacktestConfig, Backtest engine: conviction sizing, vol targeting, leverage cap, costs.  Mechani, proba_up: model P(up) indexed by date; daily_returns: same calendar., proba_up: model P(up) indexed by date; daily_returns: same calendar., Same execution/cost mechanics for a strategy that emits target weights     direc, run_backtest(), run_weights_backtest(), load_market() (+23 more)

### Community 32 - "Community 32"
Cohesion: 0.14
Nodes (12): 1. Test suite (fast, always first), 2. Scorecard reproduction (the numbers are the product), 3. Experiment scripts regenerate their published tables, 4. Dashboard boots with the REAL banner, 5. Data lineage, 5. Honesty checklist (read RESULTS_REAL.md ledger line), 6. Honesty checklist (read RESULTS_REAL.md ledger line), code:bash (.venv/bin/python -m pytest tests/ -q) (+4 more)

### Community 33 - "Community 33"
Cohesion: 0.25
Nodes (6): 1. Pre-register before running, 2. Leakage test first, feature second, 3. Evaluate in the standard vehicle, 4. Grow the ledger — even for failures, 5. Report as found, Adding a trial without lying to yourself

### Community 35 - "Community 35"
Cohesion: 0.14
Nodes (20): annual_features(), build_gtm_features(), load_annual_kg(), Guatemala cardamom export volume — Banco de Guatemala primary source.  UN Comtra, Annual export volume in kg, indexed by calendar year., Per-year features, using only same-or-prior years per row.      gtm_vol_yoy, Step-function daily features under the 01-Apr-(Y+1) publication rule., Feature block on the market calendar (ready for build_features alt=). (+12 more)

### Community 37 - "Community 37"
Cohesion: 0.36
Nodes (8): _crop_year(), _day_in_crop_year(), hurst_rs(), inventory_overhang(), Auction-physics features — signal mined from columns the pipeline never touched,, R/S Hurst exponent estimate (three-scale log-log slope)., Crop-year label: Aug-2019..Jul-2020 -> 2019., log(crop-year-to-date arrivals / past-years' norm at the same point).      Posit

### Community 38 - "Community 38"
Cohesion: 0.16
Nodes (12): mean_abs_shap(), SHAP interpretability — per-prediction attribution on top of the permutation imp, SHAP values for a fitted model from src/models/baselines.py.      logistic  -> L, Global importance: mean |SHAP| per feature, descending., shap_report(), shap_values_for(), make_gbm(), make_logistic() (+4 more)

### Community 39 - "Community 39"
Cohesion: 0.52
Nodes (5): fetch(), idukki_5d_forecast(), main(), GEFS forecast-rain pilot — proves the keyless path before any backfill.      pyt, Sum of 3-hourly APCP over leads 0-120h, area-meaned over the box.

### Community 44 - "Community 44"
Cohesion: 0.24
Nodes (11): canon_auctioneer(), main(), Extend the spot backbone with new auction days — THE lever.      python scripts/, Match keys across the site's commas and the repaired dump's semicolons.      The, rebuild_market(), _page(), Tests for archive-page parsing across markup eras + refresh key matching (offlin, The Jul-2026 site era: header cells are <td>, read_html sees ints. (+3 more)

### Community 52 - "Community 52"
Cohesion: 0.48
Nodes (6): already_done(), auction_dates(), fetch(), idukki_5d_forecast(), main(), GEFS reforecast-era forecast-rain backfill (2014-11-07 -> 2019-12-31).      pyth

## Knowledge Gaps
- **187 isolated node(s):** `Mutating TODAY's auction must not change TODAY's features.`, `The synthetic extension must not contaminate real rows' features.`, `Mutating TODAY's auction must not change TODAY's features.`, `Mutating TODAY's as-issued forecast must not change TODAY's feature.      The ra`, `The shift is a real shift, not an accidental no-op: mutating day t's     forecas` (+182 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **10 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `parse_oni()` connect `Community 3` to `Community 0`, `Community 44`?**
  _High betweenness centrality (0.149) - this node is a cross-community bridge._
- **Why does `build_features()` connect `Community 0` to `Community 7`, `Community 31`?**
  _High betweenness centrality (0.103) - this node is a cross-community bridge._
- **Why does `main()` connect `Community 0` to `Community 3`, `Community 6`, `Community 31`?**
  _High betweenness centrality (0.086) - this node is a cross-community bridge._
- **Are the 16 inferred relationships involving `build_features()` (e.g. with `main()` and `run_cv()`) actually correct?**
  _`build_features()` has 16 INFERRED edges - model-reasoned connections that need verification._
- **Are the 16 inferred relationships involving `BacktestConfig` (e.g. with `main()` and `run_cv()`) actually correct?**
  _`BacktestConfig` has 16 INFERRED edges - model-reasoned connections that need verification._
- **Are the 14 inferred relationships involving `run_backtest()` (e.g. with `main()` and `run_cv()`) actually correct?**
  _`run_backtest()` has 14 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Mutating TODAY's auction must not change TODAY's features.`, `The synthetic extension must not contaminate real rows' features.`, `Mutating TODAY's auction must not change TODAY's features.` to the rest of the system?**
  _187 weakly-connected nodes found - possible documentation gaps or missing edges._