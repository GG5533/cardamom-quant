# Data status — what's real, what's pending, and why

**As of:** July 7, 2026

## UPDATE 07-Jul-2026: full backfill COMPLETE (via browser crawl)

The entire archive was crawled through the user's own browser (same-origin
fetch, 220ms delays, zero errors): **568 pages → 5,671 sessions → 5,655
repaired rows → 3,148 auction days, 07-Nov-2014 → 07-Jul-2026**, now in
`data/processed/market.parquet` (+ `spot_daily_full.csv`,
`sessions_full_repaired.csv`). 15 malformed source rows (Indian-lakh comma
grouping, one typo) are quarantined in
`data/raw/browser/sessions_repaired.csv.rejected` — rejected, never imputed.
A raw copy also sits in your Downloads (`spices_sessions_full.csv`).
Note: the site's thousands-separator commas corrupt naive parsers — the
repair logic lives in `scripts/repair_sessions.py`-equivalent (see git
history) and validates each row against a unique sane re-grouping.

**Real out-of-sample results from this dataset: see `RESULTS_REAL.md`.**
Still pending: IMD rain (local imdlib run), MCX Bhavcopy (manual download —
the browser extension rightly blocks automated control of exchange sites),
DMI/FX/Comtrade fetchers (code ready, wire when wanted).

---

*Original status below (superseded):*

## Real data in the repo now (`data/`)

| Dataset | Coverage | Source | Files |
|---|---|---|---|
| Spot auctions (sessions + daily) | **19 Jun → 2 Jul 2026**, 20 sessions / 11 auction days | indianspices.com archive, fetched live 02-Jul-2026 | `data/raw/spices_board/page_000*.html`, `data/processed/spices_sessions_real.csv`, `spot_daily_real.csv` |
| ONI (ENSO index) | **Jan 2010 → Feb 2026**, 194 months | NOAA CPC `oni.ascii.txt`, fetched live 02-Jul-2026 | `data/raw/climate/oni.ascii.txt`, `data/processed/oni_monthly_real.csv` |
| Joined sample (spot + ONI + calendars) | 11 days | derived | `data/processed/market_sample_real.csv` |

Live snapshot worth knowing: spot averaged ₹2,763–3,005/kg over the window
(Jul 2 print ₹3,005, the local high), ~2 sessions/day, 40–129 t arriving per
auction day with ~95% sell-through — demand tension is currently high.
Latest ONI anomaly −0.16 (neutral, slightly cool). Full archive spans
**5,671 session records (568 pages)** — a decade of history awaiting backfill.

Note on raw pages: the sandbox can only retrieve pages through a rendering
fetcher, so the raw files are minimal HTML reconstructions of the fetched
tables (provenance comment in each file). Your local backfill replaces them
with byte-original pages.

## Why the rest isn't here yet — and the one command that finishes it

This session's compute environment routes all traffic through an egress
allowlist that blocks indianspices.com, NOAA, FRED, IMD and UN Comtrade from
scripts (only the interactive fetcher could reach them, which can't run a
567-page backfill). Nothing about the code is blocked — it's environment
policy. On your Mac:

```bash
cd cardamom-quant
pip install -r requirements-data.txt scikit-learn
python scripts/build_dataset.py --refresh     # ~20 min: full auction archive + IMD rain
python run.py                                  # real scorecard
```

Then per feed, in priority order:

1. **Spices Board (automatic):** `--refresh` walks all 568 pages politely
   (1.5 s delay) and is incremental afterwards — daily top-ups fetch ~1 page.
2. **IMD rainfall (automatic):** downloaded by the same command via imdlib.
3. **MCX Bhavcopy (manual, ~230 files):** download daily CSVs since
   2025-07-29 from mcxindia.com/market-data/bhavcopy into `data/raw/mcx/`.
   The parser handles both file eras and builds the continuous series.
4. **ONI/DMI/FX/Comtrade (automatic):** small fetchers in
   `src/data/climate_indices.py`, `fx.py`, `comtrade.py` — wire into
   `scripts/build_dataset.py` when you want them in the feature matrix
   (ONI already has real data in the repo).

## Verified behaviors from today's real-data run

The pipeline ran on the real 11-day sample end-to-end: parse → dedupe (the
archive's page-boundary duplicate row was correctly dropped) → qty-weighted
daily aggregation → validation passed → climate join correctly returned NaN
for ONI on June/July trading days (the 2-month publication lag refusing to
leak the JFM print forward — the guard working, not a bug).
