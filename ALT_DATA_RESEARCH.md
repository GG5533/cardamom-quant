# Cardamom Quant — Alternative-Data Research (beyond the obvious)

**Date:** July 2, 2026 · **Status:** two signal families implemented + tested; roadmap ranked by leverage

The original signal set (seasonality + realized rain + basis) is what any
competent analyst would build. This document is the out-of-the-box layer: what
actually moves cardamom that isn't in the price history, ranked by evidence,
feasibility, and leakage risk.

---

## Implemented today (code + 8 new tests, all offline)

### 1. Moving demand calendars — `src/data/calendars.py`

**The insight:** the model's day-of-year sin/cos seasonality is structurally
blind to the strongest demand seasonal in this market, because that seasonal
*moves*. Saudi Arabia and the Gulf are the dominant export market (gahwa
culture), Gulf retailers stock **6–8 weeks before Ramadan**, grocery spend
rises ~30% in the two weeks prior ([Middle East Insider](https://themiddleeastinsider.com/2026/03/16/ramadan-gulf-economies/),
[Gulf News](https://gulfnews.com/world/gulf/saudi/ramadan-2025-surge-in-demand-for-culinary-items-in-saudi-arabia-2-1.500060724)),
and Saudi import prices nearly doubled into the 2024 Ramadan–Hajj window
([Selina Wamucii](https://www.selinawamucii.com/insights/prices/saudi-arabia/cardamom/)).
Ramadan drifts ~11 days earlier every Gregorian year — an 11-day/year rotating
seasonal that day-of-year features cannot represent. Same logic domestically
for Diwali (mid-Oct to mid-Nov oscillation).

**Features:** `days_to_ramadan`, `ramadan_stocking` (14–56d window),
`ramadan_proximity` (graded 90d ramp), `days_to_diwali`, `diwali_window`.
Tabulated dates 2010–2028; ±1-day moon-sighting error is immaterial against
multi-week windows by design.

**Leakage note (a showcase line):** calendars are the only "future" input the
model is allowed — they're known ex ante with certainty. The test suite
includes a test proving the stocking window lands in *different months* in
different years.

### 2. Climate teleconnections — `src/data/climate_indices.py`

**The insight:** the rainfall loader measures weather that already happened;
ENSO and the Indian Ocean Dipole *lead* the Indian monsoon by months (El Niño
suppresses, positive IOD enhances — [NOAA climate.gov](https://www.climate.gov/news-features/blogs/enso/meet-enso%E2%80%99s-neighbor-indian-ocean-dipole),
[Down To Earth on the 2026 El Niño monsoon outlook](https://www.downtoearth.org.in/climate-change/monsoon-2026-has-arrived-but-indias-rain-season-begins-under-el-ni%C3%B1o-shadow)).
For a crop priced off Kerala yield expectations, that upgrades weather from a
lagging to a *forecastable* driver.

**Data (verified):** NOAA CPC [ONI](https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt)
— fetched 2026-07-02, plain text, 1950→JFM-2026. NOAA PSL
[DMI](https://psl.noaa.gov/gcos_wgsp/Timeseries/Data/dmi.had.long.data) —
classic PSL layout, parser handles sentinels.

**Leakage discipline (the differentiator):** ONI's 3-month centred season is
only fully published ~2 months after its centre month, so features are
**publication-lagged** (2m ONI, 1m DMI) — and there's a unit test asserting a
February trading day cannot see the DJF value. Most Kaggle-grade projects get
exactly this wrong.

---

## Implemented round 2 (code + 9 more tests)

**3. Guatemala supply shock — `src/data/comtrade.py`.** Guatemala is the
world's #1 exporter (~52% of 2024 export value, [WorldsTopExports](https://www.worldstopexports.com/top-cardamoms-exports-imports-by-country-plus-average-prices/));
its 2024-25 crop collapsed 40–50% to ~17k t and Indian exports are projected
to roughly double into the vacuum ([India Seatrade News](https://indiaseatradenews.com/indias-cardamom-exports-gain-as-guatemala-production-falters/),
[IREF](https://iref.net/news/indian-cardamom-exporters-eye-record-cardamom-exports)).
Loader hits [UN Comtrade v1](https://github.com/uncomtrade/comtradeapicall)
(free tier: keyless, 500 records/call; free key lifts limits, env
`COMTRADE_KEY`) for monthly HS 0908.31/32 exports; features `gtm_exp_kg_yoy`
and `gtm_supply_shock` (negative yoy z-score — positive when the crop fails),
mapped to trading days with a **3-month publication lag**. A unit test
reproduces the 2024-25 collapse and asserts the shock gauge fires.

**4. FX competitiveness — `src/data/fx.py`.** USD/INR daily from FRED's
keyless CSV endpoint (`fredgraph.csv?id=DEXINUS`, verified reachable);
features: level, 21d momentum, 63d z-score. USD/GTQ deliberately cut (no
free daily series; Comtrade volumes capture the Guatemala effect directly) —
scope cuts with reasons are part of the showcase.

**5. Auction microstructure — `src/features/alt_features.py`.** The
zero-cost signal that was already in hand: `tension` (sold/arrived,
EW-smoothed — do buyers absorb everything or walk away), `tension_z_63`,
and `arrivals_surprise` (log arrivals vs trailing normal — real-time supply
surprise). All shifted one day; a dedicated test mutates today's auction
print and asserts today's features don't move — strict causality, proven.

**Assembly:** `build_alt_features()` returns a stable 17-column schema on
the spot calendar regardless of which feeds are wired (missing = explicit
NaN, never silent zeros); append `ALT_FEATURE_COLS` to FEATURE_COLS and
re-run the walk-forward.

## Remaining roadmap

**6. Stretch (flagship visual, not alpha-critical):** Sentinel-2 NDVI over
the Idukki belt as a canopy-health proxy (Google Earth Engine, free tier);
Google Trends "cardamom price" as retail-attention proxy. Effort: high/low;
both make excellent dashboard panels even if features wash out.

**Rejected:** scraping broker price pages (ToS, no integrity), paid
alternative-data feeds (defeats the reproducibility story).

---

## Why this layer wins interviews

The narrative arc becomes: *price seasonality → real weather → weather you can
forecast (ENSO/IOD) → demand calendars that rotate against the solar year →
global supply shocks (Guatemala)* — each step adds an orthogonal information
source with an explicit publication-lag policy and a test proving no
look-ahead. That's a desk research process, not a notebook.

All five signal families are now implemented with 17 offline tests across
`tests/test_alt_features.py` and `tests/test_signal_layer.py` (28 total in
the suite). Next: append `ALT_FEATURE_COLS` to FEATURE_COLS in
`features/engineering.py`, re-run the purged walk-forward on real spot data,
and report which alt features earn their place — including the ones that
don't.
