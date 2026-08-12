# Graph Report - cardamom-quant  (2026-08-11)

## Corpus Check
- 83 files · ~173,977 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 851 nodes · 1605 edges · 89 communities (74 shown, 15 thin omitted)
- Extraction: 85% EXTRACTED · 15% INFERRED · 0% AMBIGUOUS · INFERRED: 239 edges (avg confidence: 0.79)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `17881e2b`
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
- [[_COMMUNITY_Community 55|Community 55]]
- [[_COMMUNITY_Community 56|Community 56]]
- [[_COMMUNITY_Community 57|Community 57]]
- [[_COMMUNITY_Community 58|Community 58]]
- [[_COMMUNITY_Community 59|Community 59]]
- [[_COMMUNITY_Community 60|Community 60]]
- [[_COMMUNITY_Community 61|Community 61]]
- [[_COMMUNITY_Community 62|Community 62]]
- [[_COMMUNITY_Community 63|Community 63]]
- [[_COMMUNITY_Community 64|Community 64]]
- [[_COMMUNITY_Community 65|Community 65]]
- [[_COMMUNITY_Community 66|Community 66]]
- [[_COMMUNITY_Community 67|Community 67]]
- [[_COMMUNITY_Community 68|Community 68]]
- [[_COMMUNITY_Community 78|Community 78]]
- [[_COMMUNITY_Community 79|Community 79]]
- [[_COMMUNITY_Community 80|Community 80]]
- [[_COMMUNITY_Community 81|Community 81]]
- [[_COMMUNITY_Community 82|Community 82]]
- [[_COMMUNITY_Community 83|Community 83]]
- [[_COMMUNITY_Community 84|Community 84]]
- [[_COMMUNITY_Community 85|Community 85]]
- [[_COMMUNITY_Community 86|Community 86]]
- [[_COMMUNITY_Community 87|Community 87]]
- [[_COMMUNITY_Community 88|Community 88]]

## God Nodes (most connected - your core abstractions)
1. `build_features()` - 23 edges
2. `REAL out-of-sample results — July 7, 2026` - 23 edges
3. `REAL out-of-sample results — July 7, 2026` - 22 edges
4. `PurgedWalkForward` - 20 edges
5. `BacktestConfig` - 19 edges
6. `run_backtest()` - 19 edges
7. `main()` - 18 edges
8. `SpicesBoardLoader` - 18 edges
9. `build_physics_features()` - 17 edges
10. `deflated_sharpe()` - 16 edges

## Surprising Connections (you probably didn't know these)
- `test_parse_comtrade_sums_hs_codes()` --calls--> `parse_comtrade()`  [INFERRED]
  tests/test_signal_layer.py → /Users/samihabbal/Documents/Claude/Projects/My personal Profile and who i am/cardamom-quant/src/data/comtrade.py
- `test_microstructure_is_strictly_causal()` --calls--> `auction_microstructure()`  [INFERRED]
  tests/test_signal_layer.py → /Users/samihabbal/Documents/Claude/Projects/My personal Profile and who i am/cardamom-quant/src/features/alt_features.py
- `test_tension_bounded_and_sane()` --calls--> `auction_microstructure()`  [INFERRED]
  tests/test_signal_layer.py → /Users/samihabbal/Documents/Claude/Projects/My personal Profile and who i am/cardamom-quant/src/features/alt_features.py
- `test_build_alt_features_schema_stable_without_optional_feeds()` --calls--> `build_alt_features()`  [INFERRED]
  tests/test_signal_layer.py → /Users/samihabbal/Documents/Claude/Projects/My personal Profile and who i am/cardamom-quant/src/features/alt_features.py
- `test_build_alt_features_with_climate()` --calls--> `build_alt_features()`  [INFERRED]
  tests/test_signal_layer.py → /Users/samihabbal/Documents/Claude/Projects/My personal Profile and who i am/cardamom-quant/src/features/alt_features.py

## Communities (89 total, 15 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.32
Nodes (9): ablation_table(), deflated_sharpe(), expected_max_sharpe_annual(), probabilistic_sharpe(), Robustness statistics — is the Sharpe distinguishable from luck?  Three tools mo, results: {variant_name: metrics_dict} -> tidy comparison frame., PSR: P(true SR > benchmark), adjusting for skew/kurtosis (B&LdP 2012).      Comp, E[max SR] under n_trials of zero-true-skill strategies whose SR     estimates ha (+1 more)

### Community 1 - "Community 1"
Cohesion: 0.12
Nodes (29): build_physics_features(), _crop_year(), _day_in_crop_year(), hurst_rs(), inventory_overhang(), Auction-physics features — signal mined from columns the pipeline never touched,, Assemble the block on the market calendar; all columns lag-safe., R/S Hurst exponent estimate (three-scale log-log slope). (+21 more)

### Community 2 - "Community 2"
Cohesion: 0.1
Nodes (34): brier_score(), calibration_summary(), calibration_table(), enso_phase(), isotonic_calibrator(), Probability calibration + regime-conditional performance.  Calibration: a 0.65 f, Fit an isotonic map raw p -> calibrated p on a held-out slice.      The slice mu, elnino' (>= +0.5), 'lanina' (<= -0.5), else 'neutral'. (+26 more)

### Community 3 - "Community 3"
Cohesion: 0.09
Nodes (31): add_calendar_features(), _days_to_next(), Demand-calendar features — the moving seasonals that day-of-year misses.  The ou, Signed day count to the next occurrence of an annual event.      Positive = even, Feature block for a given trading calendar. All ex-ante knowable., fetch_dmi(), fetch_oni(), parse_oni() (+23 more)

### Community 4 - "Community 4"
Cohesion: 0.19
Nodes (11): ABC, BaseLoader, fetch(), parse(), BaseLoader: the common contract every real-data feed implements.  Design princip, Raised when a loader's output violates its schema contract., Fetch → parse → validate → load, with an immutable raw cache., Generic schema checks; subclasses add source-specific rules. (+3 more)

### Community 5 - "Community 5"
Cohesion: 0.31
Nodes (14): _mcx_loader(), Unit tests for the real-data ingestion layer.  Everything here runs offline agai, On a roll day the spliced return must be the NEW contract's own     day-over-day, test_build_market_dataset_alignment(), test_mcx_continuous_level_no_roll_jump(), test_mcx_front_selection_and_spliced_return(), test_mcx_parses_both_schema_eras(), test_mcx_regime_guard() (+6 more)

### Community 6 - "Community 6"
Cohesion: 0.08
Nodes (43): fetch_usdinr(), parse_fred_csv(), FX competitiveness — USD/INR from FRED (no API key needed).  Indian cardamom com, fredgraph.csv -> daily Series; '.' means missing., FX features on the trading calendar (ffill weekends/holidays, max 5d)., to_daily_features(), build_forecast_rain_features(), load_forecast_rain() (+35 more)

### Community 7 - "Community 7"
Cohesion: 0.13
Nodes (23): _canon(), _chain_hash(), ChainedCsv, forecast_ledger(), outcome_ledger(), Tamper-evident forecast ledger — the prospective-validation backbone.  A backtes, Append-only CSV where every row extends a hash chain., Recompute every hash; raise on the first broken link. (+15 more)

### Community 8 - "Community 8"
Cohesion: 0.13
Nodes (13): 1. The critical finding (this reshapes the project — for the better), 2.1 Spices Board e-auction (spot) — PRIMARY, verified working, 2.2 MCX futures (Bhavcopy) — the loader Step 1 names, 2.3 Rainfall (IMD) — the weather signal, 2.4 Rejected alternatives (document these — rejection rationale is showcase material), 2. Data source research (verified where possible), 3. `loaders.py` design, 4. Build sequence (with acceptance criteria) (+5 more)

### Community 9 - "Community 9"
Cohesion: 0.25
Nodes (12): capacity_report(), market_capacity(), participation_table(), Capacity analysis — could this strategy actually be run, and at what size?  A Sh, Recent tradable-size envelope from real market data., For each assumed capital: turnover-driven daily participation in the     auction, Tests for the capacity module (offline)., test_capacity_report_renders() (+4 more)

### Community 10 - "Community 10"
Cohesion: 0.2
Nodes (8): code:bash (cd cardamom-quant), First 30 minutes in VS Code, Gotchas / tribal knowledge, HANDOFF — cardamom-quant → VS Code, One-line status if anyone asks, Priority queue (in order of expected value), State: what is DONE, What this is

### Community 11 - "Community 11"
Cohesion: 0.22
Nodes (7): 1. Moving demand calendars — `src/data/calendars.py`, 2. Climate teleconnections — `src/data/climate_indices.py`, Cardamom Quant — Alternative-Data Research (beyond the obvious), Implemented round 2 (code + 9 more tests), Implemented today (code + 8 new tests, all offline), Remaining roadmap, Why this layer wins interviews

### Community 12 - "Community 12"
Cohesion: 0.2
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
Cohesion: 0.64
Nodes (5): join_ok(), main(), partitions(), Repair thousands-separator commas in the browser-crawled sessions CSV.  Each dat, sane()

### Community 17 - "Community 17"
Cohesion: 0.04
Nodes (46): — and cut, and cut, called a tie, Dataset v1.1 (15-Jul-2026): the lever, pulled — and a lesson in, estimator variance, Forecast-rain groundwork (pilot passed), PROSPECTIVE VALIDATION (live since 16-Jul-2026), REAL out-of-sample results — July 7, 2026 (+38 more)

### Community 31 - "Community 31"
Cohesion: 0.1
Nodes (32): BacktestConfig, Backtest engine: conviction sizing, vol targeting, leverage cap, costs.  Mechani, proba_up: model P(up) indexed by date; daily_returns: same calendar., proba_up: model P(up) indexed by date; daily_returns: same calendar., Same execution/cost mechanics for a strategy that emits target weights     direc, run_backtest(), run_weights_backtest(), load_market() (+24 more)

### Community 32 - "Community 32"
Cohesion: 0.13
Nodes (12): 1. Test suite (fast, always first), 2. Scorecard reproduction (the numbers are the product), 3. Experiment scripts regenerate their published tables, 4. Dashboard boots with the REAL banner, 5. Data lineage, 5. Honesty checklist (read RESULTS_REAL.md ledger line), 6. Honesty checklist (read RESULTS_REAL.md ledger line), code:bash (.venv/bin/python -m pytest tests/ -q) (+4 more)

### Community 33 - "Community 33"
Cohesion: 0.22
Nodes (6): 1. Pre-register before running, 2. Leakage test first, feature second, 3. Evaluate in the standard vehicle, 4. Grow the ledger — even for failures, 5. Report as found, Adding a trial without lying to yourself

### Community 35 - "Community 35"
Cohesion: 0.18
Nodes (19): annual_features(), build_gtm_features(), load_annual_kg(), Guatemala cardamom export volume — Banco de Guatemala primary source.  UN Comtra, Annual export volume in kg, indexed by calendar year., Per-year features, using only same-or-prior years per row.      gtm_vol_yoy, Step-function daily features under the 01-Apr-(Y+1) publication rule., Feature block on the market calendar (ready for build_features alt=). (+11 more)

### Community 37 - "Community 37"
Cohesion: 0.21
Nodes (10): aggregate_daily(), _normalise(), _parse_html(), SpicesBoardLoader — small-cardamom e-auction archive (the spot backbone).  The S, Session-level table across all cached pages, deduped., Daily aggregate: quantity-weighted average price + supply columns., Session-level table across all cached pages, deduped., Daily aggregate: quantity-weighted average price + supply columns. (+2 more)

### Community 38 - "Community 38"
Cohesion: 0.29
Nodes (5): make_gbm(), make_logistic(), Models. The rule of this repo: ML must beat the dumb seasonal rule or we say so, Probability 0.5 +/- edge by season. No fitting on prices at all., SeasonalBaseline

### Community 39 - "Community 39"
Cohesion: 0.54
Nodes (5): fetch(), idukki_5d_forecast(), main(), GEFS forecast-rain pilot — proves the keyless path before any backfill.      pyt, Sum of 3-hourly APCP over leads 0-120h, area-meaned over the box.

### Community 44 - "Community 44"
Cohesion: 0.25
Nodes (11): canon_auctioneer(), main(), Extend the spot backbone with new auction days — THE lever.      python scripts/, Match keys across the site's commas and the repaired dump's semicolons.      The, rebuild_market(), _page(), Tests for archive-page parsing across markup eras + refresh key matching (offlin, The Jul-2026 site era: header cells are <td>, read_html sees ints. (+3 more)

### Community 52 - "Community 52"
Cohesion: 0.61
Nodes (6): already_done(), auction_dates(), fetch(), idukki_5d_forecast(), main(), GEFS reforecast-era forecast-rain backfill (2014-11-07 -> 2019-12-31).      pyth

### Community 55 - "Community 55"
Cohesion: 0.14
Nodes (13): 1. The critical finding (this reshapes the project — for the better), 2.1 Spices Board e-auction (spot) — PRIMARY, verified working, 2.2 MCX futures (Bhavcopy) — the loader Step 1 names, 2.3 Rainfall (IMD) — the weather signal, 2.4 Rejected alternatives (document these — rejection rationale is showcase material), 2. Data source research (verified where possible), 3. `loaders.py` design, 4. Build sequence (with acceptance criteria) (+5 more)

### Community 56 - "Community 56"
Cohesion: 0.35
Nodes (11): filter_anomaly(), fit_params(), _fourier(), half_life_days(), kalman_features(), KalmanParams, kappa(), Structural decomposition of log spot: level + seasonal + AR(1) anomaly.  Ported/ (+3 more)

### Community 57 - "Community 57"
Cohesion: 0.18
Nodes (6): add_anomaly(), IMDRainfallLoader, IMDRainfallLoader — Idukki cardamom-belt rainfall with anomaly features.  Histor, Download gridded rain via imdlib into raw_dir (idempotent)., Gridded binary -> Idukki daily area-mean -> anomaly features., Real-data ingestion layer for cardamom-quant.

### Community 58 - "Community 58"
Cohesion: 0.22
Nodes (8): code:bash (cd cardamom-quant), First 30 minutes in VS Code, Gotchas / tribal knowledge, HANDOFF — cardamom-quant → VS Code, One-line status if anyone asks, Priority queue (in order of expected value), State: what is DONE, What this is

### Community 59 - "Community 59"
Cohesion: 0.25
Nodes (7): BaseLoader, add_features(), GuatemalaExportsLoader, parse_comtrade(), Guatemala cardamom exports via UN Comtrade — the cross-market supply signal.  Wh, Comtrade JSON -> monthly frame [exp_kg, exp_usd], HS codes summed., to_daily_features()

### Community 60 - "Community 60"
Cohesion: 0.25
Nodes (7): 1. Moving demand calendars — `src/data/calendars.py`, 2. Climate teleconnections — `src/data/climate_indices.py`, Cardamom Quant — Alternative-Data Research (beyond the obvious), Implemented round 2 (code + 9 more tests), Implemented today (code + 8 new tests, all offline), Remaining roadmap, Why this layer wins interviews

### Community 61 - "Community 61"
Cohesion: 0.29
Nodes (6): code:bash (cd cardamom-quant), Data status — what's real, what's pending, and why, Real data in the repo now (`data/`), UPDATE 07-Jul-2026: full backfill COMPLETE (via browser crawl), Verified behaviors from today's real-data run, Why the rest isn't here yet — and the one command that finishes it

### Community 62 - "Community 62"
Cohesion: 0.43
Nodes (6): fetch_expiry(), main(), MCX cardamom Bhavcopy backfill via the real Bhav Copy UI (Commodity Wise tab)., Post-relaunch expiries, taken as the contiguous prefix of the     dropdown befor, relevant_expiries(), set_date()

### Community 63 - "Community 63"
Cohesion: 0.33
Nodes (5): Attachments (in order of impact), LinkedIn post — Cardamom Quant, Main post, Posting notes, Short variant (for comments / reposts)

### Community 64 - "Community 64"
Cohesion: 0.33
Nodes (5): Cardamom Quant, Honesty guardrails, Layout, Run, The data story (the interesting part)

### Community 65 - "Community 65"
Cohesion: 0.29
Nodes (9): calibrated_fold_proba(), evaluate_stream(), main(), Edge hunt, round 2 — four pre-registered trials, counted before running.      py, Per-fold isotonic-calibrated GBM stream for a fold-varying X builder.      X may, main(), _naive_full_history_diagnostic(), MCX futures basis, ONE pre-registered trial — T14.      python scripts/mcx_basis (+1 more)

### Community 66 - "Community 66"
Cohesion: 0.67
Nodes (3): load_market(), Cardamom Quant — interactive dashboard.      pip install streamlit     streamlit, run_cv()

### Community 67 - "Community 67"
Cohesion: 0.67
Nodes (3): load_dataset(), main(), cardamom-quant — end-to-end run.      python run.py                 # real data

### Community 68 - "Community 68"
Cohesion: 0.17
Nodes (12): build_continuous(), _map_columns(), MCXBhavcopyLoader, _norm(), MCXBhavcopyLoader — cardamom futures from MCX Bhavcopy files.  Tier-1 strategy (, Tier-1 is manual: files are dropped into raw_dir by the user.          This just, Read a CSV or a zip containing CSVs., All cardamom contract-day rows across all cached files. (+4 more)

### Community 78 - "Community 78"
Cohesion: 0.26
Nodes (8): load_dataset(), main(), cardamom-quant — end-to-end run.      python run.py                 # real data, build_features(), forward_returns(), Leakage-safe feature engineering.  Contract: every feature value dated t uses on, market: aligned frame from build_market_dataset (or synthetic clone).      Retur, The realized forward return the backtest trades against.

### Community 79 - "Community 79"
Cohesion: 0.35
Nodes (9): block_bootstrap_sharpe(), Circular block bootstrap of the annualized Sharpe ratio., gbm_probability_streams(), grid_cell(), main(), Horizon-matched trading experiment — the make-or-break test.      python scripts, OOS P(up) per fold: raw (full-train fit, the published pipeline) and     calibra, Staggered implementation: 1/reb of the book rebalances each day.      A single 5 (+1 more)

### Community 80 - "Community 80"
Cohesion: 0.24
Nodes (4): main(), Slicing-robust champion estimate — kill the fold-boundary wobble.      python sc, PurgedWalkForward, Purged + embargoed expanding-window walk-forward CV.  Why purging matters here:

### Community 81 - "Community 81"
Cohesion: 0.33
Nodes (6): load_dataset(), main(), backtest_metrics(), classification_metrics(), format_scorecard(), Scorecard metrics — reported honestly, baseline included.

### Community 82 - "Community 82"
Cohesion: 0.53
Nodes (6): build_market_dataset(), load_futures(), load_rain(), load_spot(), Real-data facade — the swap-in point that replaces synthetic.py.  Public API (wh, Join the three feeds on a master calendar, compute basis honestly.

### Community 83 - "Community 83"
Cohesion: 0.52
Nodes (4): main(), member_matrices(), Edge hunt, round 6 — the probability ensemble, ONE pre-registered trial.      py, The three member designs: (X, y) each, on their own market variant.

## Knowledge Gaps
- **234 isolated node(s):** `Mutating TODAY's auction must not change TODAY's features.`, `The synthetic extension must not contaminate real rows' features.`, `Mutating TODAY's auction must not change TODAY's features.`, `Real-shaped spot+futures frame (as build_market_dataset would join     them, pos`, `Mutating TODAY's basis (as build_market_dataset would compute it from     a chan` (+229 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **15 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `build_features()` connect `Community 78` to `Community 65`, `Community 2`, `Community 6`, `Community 7`, `Community 79`, `Community 80`, `Community 81`, `Community 83`, `Community 85`, `Community 86`, `Community 87`, `Community 88`, `Community 31`?**
  _High betweenness centrality (0.119) - this node is a cross-community bridge._
- **Why does `parse_oni()` connect `Community 3` to `Community 81`, `Community 44`, `Community 78`?**
  _High betweenness centrality (0.099) - this node is a cross-community bridge._
- **Why does `main()` connect `Community 87` to `Community 0`, `Community 1`, `Community 65`, `Community 35`, `Community 78`, `Community 80`?**
  _High betweenness centrality (0.059) - this node is a cross-community bridge._
- **Are the 19 inferred relationships involving `build_features()` (e.g. with `main()` and `run_cv()`) actually correct?**
  _`build_features()` has 19 INFERRED edges - model-reasoned connections that need verification._
- **Are the 15 inferred relationships involving `PurgedWalkForward` (e.g. with `main()` and `run_cv()`) actually correct?**
  _`PurgedWalkForward` has 15 INFERRED edges - model-reasoned connections that need verification._
- **Are the 16 inferred relationships involving `BacktestConfig` (e.g. with `main()` and `run_cv()`) actually correct?**
  _`BacktestConfig` has 16 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Mutating TODAY's auction must not change TODAY's features.`, `The synthetic extension must not contaminate real rows' features.`, `Mutating TODAY's auction must not change TODAY's features.` to the rest of the system?**
  _234 weakly-connected nodes found - possible documentation gaps or missing edges._