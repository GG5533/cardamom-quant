# Cardamom Quant

AI-driven price-direction model for Indian small cardamom (MCX / Spices Board
e-auctions), built with trading-desk discipline: leakage-safe features, purged
walk-forward validation, honest baselines, vol-targeted backtests after costs.

**Status:** full pipeline runs end-to-end (`python run.py`). Real-data
ingestion layer is built and tested; run `scripts/build_dataset.py --refresh`
to backfill live feeds, otherwise the calibrated `[SYNTHETIC]` testbed runs.

## The data story (the interesting part)

MCX cardamom futures were suspended in 2021 and relaunched 2025-07-29 — only
~11 months of tradable history. So the model trains on ~a decade of Spices
Board e-auction spot data (price *and* physical arrivals) and uses the young
future for basis and a tradability check. Details: `STEP1_RESEARCH_AND_PLAN.md`.

On top of the core signals sit five alt-data families (`ALT_DATA_RESEARCH.md`):
auction microstructure (demand tension, arrival surprises), rotating demand
calendars (Ramadan drifts 11 days/year against the solar calendar — invisible
to day-of-year features), ENSO/IOD monsoon teleconnections, Guatemala export
shocks (UN Comtrade), and USD/INR competitiveness. Every non-calendar feature
carries an explicit publication lag, each enforced by a unit test.

## Layout

    run.py                      # end-to-end: data → features → walk-forward → backtest
    scripts/build_dataset.py    # real-data build (Spices Board / MCX / IMD)
    src/data/                   # loaders (see src/data/README.md) + synthetic testbed
    src/features/               # engineering.py (core) + alt_features.py (alt-data)
    src/validation/walkforward.py   # purged + embargoed expanding walk-forward
    src/models/baselines.py     # seasonal rule, logistic, HistGradientBoosting
    src/backtest/engine.py      # conviction sizing, vol targeting, 15bps costs
    src/metrics.py              # scorecard
    src/analysis/               # robustness (bootstrap CI, deflated Sharpe),
                                #   calibration + ENSO/season regimes, SHAP, capacity
    scripts/analyze.py          # the full "risk committee" report
    app.py                      # Streamlit dashboard (streamlit run app.py)
    tests/                      # 49 offline tests (fixtures, no network)

## Run

    pip install -r requirements-data.txt -r requirements-analysis.txt
    python -m pytest tests/ -q        # 49 tests
    streamlit run app.py              # interactive demo (REAL/SYNTHETIC banner)
    python run.py --synthetic         # methodology testbed
    python run.py --no-alt            # ablation: core features only
    python scripts/analyze.py         # ablation x bootstrap CI x deflated Sharpe,
                                      #   calibration, regime splits, SHAP
    python scripts/build_dataset.py --refresh   # wire real feeds, then: python run.py

## Honesty guardrails

Synthetic results validate the machinery, never the market thesis — every
synthetic chart/table is tagged `[SYNTHETIC]`. The ML must beat the seasonal
baseline out-of-sample or the README says so. Costs are charged on turnover;
positions execute next-day; labels are purged from training windows.
