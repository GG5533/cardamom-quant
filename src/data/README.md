# Real-data ingestion layer

Replaces `synthetic.py` as the data source for the cardamom-quant pipeline.
See `../../STEP1_RESEARCH_AND_PLAN.md` for the full research behind these choices.

## Why the spot market is the backbone

MCX cardamom futures were **suspended in 2021 and relaunched 2025-07-29** as a
redesigned compulsory-delivery contract — only ~11 months of tradable history
exist. The Spices Board e-auction archive provides ~a decade of continuous
spot prices *and physical arrival quantities* on the same pricing basis
(ex-Kerala, Idukki) as the future. The model therefore trains on spot,
and uses futures for the basis feature and the tradability check.

## The three loaders

| Loader | Source | Fetch mode |
|---|---|---|
| `SpicesBoardLoader` | indianspices.com auction archive (paginated HTML) | automatic, incremental |
| `MCXBhavcopyLoader` | Bhavcopy files in `data/raw/mcx/` | **manual drop-folder** (tier-1 by design) |
| `IMDRainfallLoader` | IMD 0.25° gridded rain via imdlib | automatic (yearly files) |

All three follow the same contract (`base.py`): `fetch()` into an immutable
raw cache → `parse()` as a pure function of cached files → `validate()` that
fails loudly → `load()` returns the canonical frame and writes parquet.

## MCX files: what to drop where

Download daily files from <https://www.mcxindia.com/market-data/bhavcopy>
into `data/raw/mcx/`. Both eras are handled automatically:

- classic `BhavCopyDateWise_YYYYMMDD.csv`
- post-July-2024 UDiFF zips/CSVs (`TradDt`/`TckrSymb`/`XpryDt` headers)

The loader filters to CARDAMOM, picks the front contract by max open
interest, and builds the continuous series by **return splicing** (each
day's return is the front contract's own day-over-day return), so rolls
never inject phantom P&L. Returns can never straddle the 2021–2025
suspension gap — `regime` guards it and `validate()` rejects gap rows.

## Build everything

```bash
pip install -r requirements-data.txt
python scripts/build_dataset.py --refresh   # Spices Board backfill ≈ 20 min (polite delays)
python -m pytest tests/ -q                  # 11 offline tests, no network needed
```

Output: `data/processed/market.parquet` — one aligned daily frame with spot,
futures, basis (NaN where not honestly computable, with an explicit
`spot_staleness_days` column) and rainfall anomalies, ready for
`features/engineering.py`.

## Known limitations (also candidates for the LinkedIn write-up)

- Auction average price moves with grade mix, not only the market; the
  qty-weighted daily aggregate smooths across same-day sessions only.
- The gridded rain product is published with a lag; recent months need the
  district-data top-up (`source` column tracks provenance).
- Tier-2 MCX automation (the site's AJAX JSON endpoint) is deliberately not
  wired: confirm the endpoint in browser DevTools first, then add a
  `fetch()` implementation — the parser will not need to change.
