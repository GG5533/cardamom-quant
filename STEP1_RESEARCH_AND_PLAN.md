# Cardamom Quant — Step 1: Real Data Ingestion (Research & Implementation Plan)

**Date:** July 2, 2026 · **Scope:** Replace the calibrated synthetic market with real feeds via `src/data/loaders.py` · **Status:** Research complete, ready to implement

---

## 1. The critical finding (this reshapes the project — for the better)

**MCX cardamom futures were suspended in 2021 and only relaunched on July 29, 2025** after a ~4-year pause, as a redesigned compulsory-delivery contract ([Angel One](https://www.angelone.in/news/market-updates/mcx-relaunches-cardamom-futures-from-july-29-2025), [Business Standard](https://www.business-standard.com/markets/capital-market-news/mcx-launches-cardamom-futures-contracts-125072900192_1.html), [Zee Business](https://www.zeebiz.com/markets/commodities/news-mcx-cardamom-launches-futures-july-29-2025-contract-trades-at-2800-to-2935-per-kg-374944)).

Consequences for the model:

- **Only ~11 months of new-regime futures data exist** (Aug 2025 → Jul 2026). That is nowhere near enough for purged walk-forward CV on its own.
- Pre-2021 futures history exists but is a **different contract regime** separated by a 4-year gap — usable with a regime flag, not naively spliceable.
- **The Spices Board e-auction spot series is the continuous backbone.** Verified today: the archive at [indianspices.com](https://www.indianspices.com/marketing/price/domestic/daily-price-small.html) is plain server-rendered HTML, paginated (`?page=N`), **567 pages / 5,670 auction records** — roughly a decade of history, with data current through 01-Jul-2026 (avg ₹2,815–2,889/kg, ~120t arrived that day).

**Reframed thesis (and the showcase angle):** *"The tradable instrument has 11 months of history. So the signal is built and walk-forward-validated on 10+ years of cash-market e-auction data, then tested for tradability on the relaunched future via the basis."* This is exactly the kind of data-regime problem real desks handle, and it's a far stronger story than "I downloaded futures prices."

---

## 2. Data source research (verified where possible)

### 2.1 Spices Board e-auction (spot) — PRIMARY, verified working

- **URL:** `https://www.indianspices.com/marketing/price/domestic/daily-price-small.html?page=N` (N = 1…567; 10 rows/page, newest first). Fetched successfully today without JS.
- **Fields per row:** Date of Auction, Auctioneer, No. of Lots, Qty Arrived (kg), Qty Sold (kg), Max Price (₹/kg), Avg Price (₹/kg).
- **Structure notes:** ~2 auctioneer sessions per auction day (rotating licensed auctioneers: Green House, KCPMC Thekkady, Mas Enterprises, Header Systems, etc.); no auctions Sundays/holidays. There is also an [auction-details report page](https://www.indianspices.com/dailyprice-auctiondetails.html) and an on-page search form (auctioneer + date range) that POSTs — use the paginated archive for the bulk backfill, it's simpler.
- **Ingestion:** `requests` + `pandas.read_html`, 1–2 s polite delay, ~567 requests one-time backfill, then incremental top-up (fetch page 1 until overlap with cache). Dedupe on (date, auctioneer).
- **Daily series:** quantity-weighted average price across same-day sessions, plus total qty arrived/sold (a real **supply signal** the synthetic generator never had — arrivals spike in harvest season).

### 2.2 MCX futures (Bhavcopy) — the loader Step 1 names

- The [Bhavcopy page](https://www.mcxindia.com/market-data/bhavcopy) is a JS-rendered ASP.NET WebForms app — a plain GET returns an empty shell (verified today). Community downloaders drive it with Selenium; element IDs from a [working implementation](https://github.com/Khushalsawant/Download-Bhavcopy-from-MCX_India): date input `cph_InnerContainerRight_C001_txtDate_hid_val`, button `btnShowDatewise`, CSV export `cph_InnerContainerRight_C001_lnkExpToCSV`; downloaded files are named `BhavCopyDateWise_YYYYMMDD.csv`.
- The datewise grid is also populated by an AJAX JSON endpoint (`backpage.aspx/GetDateWiseBhavCopy`-style POST). **This must be confirmed in browser DevTools** (Network tab on the bhavcopy page) — endpoint names/payloads aren't documented and MCX migrated file formats to **UDiFF in July 2024** ([MCX CCL](https://www.mcxccl.com/circulars/unified-distilled-file-formats-(udiff))), so old and new files have different schemas.
- **Bhavcopy fields:** instrument, symbol, expiry, open/high/low/close, previous close, volume, value, open interest. Filter `Symbol == "CARDAMOM"`.
- **Contract specs (relaunched):** 100 kg lot, ₹/kg quotation, ex-Vandanmedu (Idukki) basis, compulsory delivery, 4 monthly expiries, DPL 4%+2%, Mon–Fri 9:00–17:00 ([Zee Business](https://www.zeebiz.com/markets/commodities/news-mcx-cardamom-launches-futures-july-29-2025-contract-trades-at-2800-to-2935-per-kg-374944), [MCX product page](https://www.mcxindia.com/products/agro-commodities/cardamom)).
- **Loader strategy — three tiers, most robust first:**
  1. **Drop-folder parser (build first, guaranteed):** parse any `BhavCopyDateWise_*.csv` / UDiFF zip placed in `data/raw/mcx/`, handling both pre- and post-UDiFF schemas. ~230 trading days since relaunch — a manual bulk download is a bounded one-time task, and the parser is the part that shows engineering discipline.
  2. **Direct AJAX fetch:** small `requests` client for the JSON endpoint once confirmed in DevTools (proper headers + ASP.NET cookies). Fast daily incremental updates.
  3. **Playwright fallback:** headless automation of the datewise export using the element IDs above, for when the endpoint changes.
- **Continuous series construction:** front contract by max volume/OI, roll on volume crossover, back-adjust (ratio) — with the roll logic unit-tested. Keep a `regime` column (`pre2021` / `post2025`); never let features cross the gap.

### 2.3 Rainfall (IMD) — the weather signal

- **Historical:** [IMDLIB](https://imdlib.readthedocs.io/en/latest/) downloads IMD Pune's gridded daily rainfall (0.25°, 1901→recent) as binary → xarray/CSV. Subset the Idukki cardamom belt (~9.4–10.2°N, 76.7–77.4°E), area-mean daily, then compute anomalies vs day-of-year climatology — exactly the `rain anomaly` feature the synthetic generator emulates, now real.
- **Recent/real-time:** IMD's [district rainfall pages](https://mausam.imd.gov.in/responsive/rainfallinformation_swd.php) and [Kerala WRIS](https://wris.kerala.gov.in/mis/wd/home/rainfall-actual) for the current-year top-up (gridded data is released with a lag).
- Note in the README: gridded product revisions mean the last few months use a different (station/district) source — flagged with a `source` column.

### 2.4 Rejected alternatives (document these — rejection rationale is showcase material)

- **Investing.com / broker sites** for futures history: ToS-restricted scraping, no OI/volume integrity, unversioned back-adjustments. Not desk-grade.
- **Agmarknet mandi prices**: cardamom coverage is patchy vs the Spices Board's own auction data (which is the primary market venue — ~all small cardamom trades through these e-auctions).
- **Paid APIs (TrueData etc.)**: real-time focus, subscription cost; overkill for EOD research.

---

## 3. `loaders.py` design

One pattern for all three feeds — this consistency is itself a showcase point:

```
BaseLoader
  .fetch(start, end)      -> raw payloads into data/raw/<source>/ (immutable cache, never re-download)
  .parse()                -> canonical tidy DataFrame
  .validate()             -> schema, monotone dates, no dupes, price sanity bounds, gap report
  .load(start, end)       -> cached-or-fetched, validated frame
```

Canonical output schemas (match `synthetic.py` so `run.py` swaps with a single flag):

| Loader | Index | Columns |
|---|---|---|
| `SpicesBoardLoader` | auction date | `spot_avg`, `spot_max`, `qty_arrived`, `qty_sold`, `n_sessions` |
| `MCXBhavcopyLoader` | trade date | `fut_close`, `fut_volume`, `fut_oi`, `contract`, `days_to_expiry`, `fut_cont` (rolled), `regime` |
| `IMDRainfallLoader` | calendar date | `rain_mm`, `rain_climatology`, `rain_anomaly`, `source` |

Implementation notes:

- **Calendar alignment is the real work:** auction days ≠ MCX trading days ≠ calendar days. Join on a master calendar; forward-fill spot for basis computation with an explicit `spot_staleness_days` column rather than silent ffill.
- **Basis features** (`basis`, `basis_z`, `basis_chg`) are only defined post-relaunch (and pre-2021). The feature matrix must carry them as missing-with-mask, and the walk-forward must not train on rows where the mask lies about availability — extend the leakage discipline you already built.
- **Target switch:** primary target becomes sign of forward 5-day **spot** return (10+ years of it); secondary evaluation re-runs the backtest on **futures** returns post-Aug-2025 as the tradability check.
- Config in one place (`config.py`): URLs, date ranges, Idukki bounding box, cost assumptions.
- Unit tests: parser fixtures (one saved HTML page, one saved Bhavcopy CSV of each schema era), roll logic, weighted-average aggregation, validation failures.

## 4. Build sequence (with acceptance criteria)

1. **`SpicesBoardLoader`** — backfill all 567 pages → parquet cache; plot the full price history. *Accept: continuous daily series, dupes = 0, matches spot-checked website values.* (~Highest value, lowest risk — do first.)
2. **`IMDRainfallLoader`** — IMDLIB backfill + anomaly computation. *Accept: monsoon seasonality visible in climatology; 2018 Kerala flood shows as extreme anomaly (natural validation).*
3. **`MCXBhavcopyLoader` tier 1** (drop-folder parser, both schemas) → then tier 2 (AJAX) if the endpoint cooperates. *Accept: post-relaunch continuous series with documented rolls; basis vs auction spot plots sane (futures ≈ spot ± carry).*
4. **Re-run the full pipeline on real data** — features → purged walk-forward → backtest vs seasonal baseline, spot target primary, futures overlay secondary. *Accept: honest results table, whatever it says.*
5. **Swap chart labels [SYNTHETIC] → [REAL]** only where the data is real; keep the synthetic run as a methodology appendix.

## 5. Showcase strategy (LinkedIn)

The post writes itself around three beats:

1. **The trap avoided:** "I went to wire real futures data and discovered the contract had only existed for 11 months — it was relaunched in July 2025 after a 4-year regulatory suspension. A backtest that spliced old and new regimes silently would be fiction."
2. **The desk-grade answer:** model the cash market (10 yr of Spices Board e-auction data — price *and* physical arrivals), validate tradability on the young future via basis. Show the [SYNTHETIC] vs [REAL] chart pair.
3. **The honest scorecard:** synthetic Sharpe 4.6 was the methodology test, not the finding; report the real out-of-sample numbers vs the seasonal baseline even if the baseline wins. That sentence is what a quant hiring manager wants to read.

Artifacts to produce for the post: one figure of the full spot history annotated with the futures suspension/relaunch window, one basis chart, one results table (baseline vs LR vs GBM, real data), repo README section "Data reality: what changed when the data got real." The Streamlit dashboard (your step 3) then demos this interactively.

## 6. Risks & mitigations

- **indianspices.com fragility** (Drupal 7, occasional downtime): immutable raw-page cache; the backfill only ever runs once.
- **MCX endpoint churn / bot-blocking:** tier-1 drop-folder parser guarantees the project never blocks on this; tiers 2–3 are conveniences.
- **Auction quality-mix shifts** (avg price moves with grade mix, not just market): note as limitation; qty-weighted averaging across same-day sessions partially smooths it.
- **11-month futures overlay is statistically thin:** present the tradability check as indicative, with wide error bars — never headline it.

---

### Sources

- [MCX Bhavcopy page](https://www.mcxindia.com/market-data/bhavcopy) · [MCX cardamom product page](https://www.mcxindia.com/products/agro-commodities/cardamom) · [MCX CCL UDiFF circular](https://www.mcxccl.com/circulars/unified-distilled-file-formats-(udiff))
- [Angel One — relaunch explainer](https://www.angelone.in/news/market-updates/mcx-relaunches-cardamom-futures-from-july-29-2025) · [Business Standard — launch note](https://www.business-standard.com/markets/capital-market-news/mcx-launches-cardamom-futures-contracts-125072900192_1.html) · [Zee Business — contract specs](https://www.zeebiz.com/markets/commodities/news-mcx-cardamom-launches-futures-july-29-2025-contract-trades-at-2800-to-2935-per-kg-374944)
- [Spices Board small cardamom auction archive](https://www.indianspices.com/marketing/price/domestic/daily-price-small.html) (fetched & verified 02-Jul-2026) · [Auction details report](https://www.indianspices.com/dailyprice-auctiondetails.html)
- [IMDLIB documentation](https://imdlib.readthedocs.io/en/latest/) · [IMD rainfall information](https://mausam.imd.gov.in/responsive/rainfallinformation_swd.php) · [Kerala WRIS rainfall](https://wris.kerala.gov.in/mis/wd/home/rainfall-actual)
- [Community MCX Selenium downloader](https://github.com/Khushalsawant/Download-Bhavcopy-from-MCX_India) (element IDs, file naming) · [Zerodha Varsity — cardamom on MCX](https://zerodha.com/varsity/chapter/cardamom-mentha-oil/)
