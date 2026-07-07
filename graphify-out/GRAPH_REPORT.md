# Graph Report - cardamom-quant  (2026-07-07)

## Corpus Check
- 49 files · ~21,423 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 349 nodes · 478 edges · 31 communities (24 shown, 7 thin omitted)
- Extraction: 78% EXTRACTED · 22% INFERRED · 0% AMBIGUOUS · INFERRED: 105 edges (avg confidence: 0.78)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `09e4de6c`
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

## God Nodes (most connected - your core abstractions)
1. `SpicesBoardLoader` - 15 edges
2. `MCXBhavcopyLoader` - 13 edges
3. `main()` - 12 edges
4. `ValidationError` - 12 edges
5. `main()` - 11 edges
6. `IMDRainfallLoader` - 11 edges
7. `main()` - 10 edges
8. `build_alt_features()` - 10 edges
9. `BaseLoader` - 10 edges
10. `run_cv()` - 9 edges

## Surprising Connections (you probably didn't know these)
- `main()` --calls--> `parse_oni()`  [INFERRED]
  run.py → src/data/climate_indices.py
- `test_parse_fred_csv_missing_dot()` --calls--> `parse_fred_csv()`  [INFERRED]
  tests/test_signal_layer.py → src/data/fx.py
- `test_fx_features_ffill_limited()` --calls--> `parse_fred_csv()`  [INFERRED]
  tests/test_signal_layer.py → src/data/fx.py
- `test_build_alt_features_schema_stable_without_optional_feeds()` --calls--> `build_alt_features()`  [INFERRED]
  tests/test_signal_layer.py → src/features/alt_features.py
- `test_build_alt_features_with_climate()` --calls--> `build_alt_features()`  [INFERRED]
  tests/test_signal_layer.py → src/features/alt_features.py

## Communities (31 total, 7 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.08
Nodes (37): BacktestConfig, Backtest engine: conviction sizing, vol targeting, leverage cap, costs.  Mechani, proba_up: model P(up) indexed by date; daily_returns: same calendar., run_backtest(), load_market(), Cardamom Quant — interactive dashboard.      pip install streamlit     streamlit, run_cv(), load_dataset() (+29 more)

### Community 1 - "Community 1"
Cohesion: 0.07
Nodes (25): ABC, BaseLoader, BaseLoader, fetch(), parse(), BaseLoader: the common contract every real-data feed implements.  Design princip, Raised when a loader's output violates its schema contract., Fetch → parse → validate → load, with an immutable raw cache. (+17 more)

### Community 2 - "Community 2"
Cohesion: 0.09
Nodes (33): brier_score(), calibration_summary(), calibration_table(), enso_phase(), Probability calibration + regime-conditional performance.  Calibration: a 0.65 f, elnino' (>= +0.5), 'lanina' (<= -0.5), else 'neutral'., regimes: frame of categorical columns on the same calendar     (e.g. enso phase,, regime_performance() (+25 more)

### Community 3 - "Community 3"
Cohesion: 0.08
Nodes (26): add_calendar_features(), _days_to_next(), Demand-calendar features — the moving seasonals that day-of-year misses.  The ou, Signed day count to the next occurrence of an annual event.      Positive = even, Feature block for a given trading calendar. All ex-ante knowable., fetch_dmi(), fetch_oni(), parse_oni() (+18 more)

### Community 4 - "Community 4"
Cohesion: 0.1
Nodes (19): build_market_dataset(), load_futures(), load_rain(), load_spot(), Real-data facade — the swap-in point that replaces synthetic.py.  Public API (wh, Join the three feeds on a master calendar, compute basis honestly., build_continuous(), _map_columns() (+11 more)

### Community 5 - "Community 5"
Cohesion: 0.1
Nodes (16): aggregate_daily(), _normalise(), _parse_html(), SpicesBoardLoader — small-cardamom e-auction archive (the spot backbone).  The S, Session-level table across all cached pages, deduped., Daily aggregate: quantity-weighted average price + supply columns., Walk the archive pages, saving each as raw HTML.          Incremental logic: the, SpicesBoardLoader (+8 more)

### Community 6 - "Community 6"
Cohesion: 0.13
Nodes (17): fetch_usdinr(), parse_fred_csv(), FX competitiveness — USD/INR from FRED (no API key needed).  Indian cardamom com, fredgraph.csv -> daily Series; '.' means missing., FX features on the trading calendar (ffill weekends/holidays, max 5d)., to_daily_features(), auction_microstructure(), Demand tension + supply surprise from qty columns. Lag-safe (shifted).      tens (+9 more)

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
Cohesion: 0.4
Nodes (4): REAL out-of-sample results — July 7, 2026, Roadmap implied by the numbers, The findings — reported as found, The scorecard (purged walk-forward, 6 folds, after 15bps costs)

## Knowledge Gaps
- **125 isolated node(s):** `cardamom-quant — end-to-end run.      python run.py                 # real data`, `Cardamom Quant — interactive dashboard.      pip install streamlit     streamlit`, `Tests for the cross-market/macro/microstructure signal layer (offline).`, `Mutating TODAY's auction must not change TODAY's features.`, `Tests for calendar and climate-index feature modules (all offline).` (+120 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **7 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `SpicesBoardLoader` connect `Community 5` to `Community 1`, `Community 3`, `Community 4`?**
  _High betweenness centrality (0.227) - this node is a cross-community bridge._
- **Why does `parse_oni()` connect `Community 3` to `Community 0`?**
  _High betweenness centrality (0.225) - this node is a cross-community bridge._
- **Why does `main()` connect `Community 3` to `Community 5`?**
  _High betweenness centrality (0.210) - this node is a cross-community bridge._
- **Are the 6 inferred relationships involving `SpicesBoardLoader` (e.g. with `BaseLoader` and `ValidationError`) actually correct?**
  _`SpicesBoardLoader` has 6 INFERRED edges - model-reasoned connections that need verification._
- **Are the 4 inferred relationships involving `MCXBhavcopyLoader` (e.g. with `BaseLoader` and `ValidationError`) actually correct?**
  _`MCXBhavcopyLoader` has 4 INFERRED edges - model-reasoned connections that need verification._
- **Are the 10 inferred relationships involving `main()` (e.g. with `parse_oni()` and `build_alt_features()`) actually correct?**
  _`main()` has 10 INFERRED edges - model-reasoned connections that need verification._
- **Are the 4 inferred relationships involving `ValidationError` (e.g. with `GuatemalaExportsLoader` and `MCXBhavcopyLoader`) actually correct?**
  _`ValidationError` has 4 INFERRED edges - model-reasoned connections that need verification._