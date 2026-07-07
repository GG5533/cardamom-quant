"""Cardamom Quant — interactive dashboard.

    pip install streamlit
    streamlit run app.py

Auto-detects real data (data/processed/market.parquet); otherwise runs the
[SYNTHETIC] methodology testbed with a banner saying exactly that. All heavy
computation is cached, so the app re-runs the walk-forward once per session.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.analysis.calibration import calibration_table, regime_performance  # noqa: E402
from src.analysis.capacity import capacity_report  # noqa: E402
from src.backtest.engine import BacktestConfig, run_backtest  # noqa: E402
from src.data import config  # noqa: E402
from src.features.alt_features import build_alt_features  # noqa: E402
from src.features.engineering import build_features  # noqa: E402
from src.metrics import backtest_metrics, classification_metrics  # noqa: E402
from src.models.baselines import MODELS  # noqa: E402
from src.validation.walkforward import PurgedWalkForward  # noqa: E402

st.set_page_config(page_title="Cardamom Quant", layout="wide")


# ----------------------------------------------------------------- data + cv
@st.cache_data(show_spinner="Loading market data…")
def load_market() -> tuple[pd.DataFrame, str]:
    real = config.PROCESSED_DIR / "market.parquet"
    if real.exists():
        return pd.read_parquet(real), "REAL"
    from src.data.synthetic import generate_market

    return generate_market(), "SYNTHETIC"


@st.cache_data(show_spinner="Running purged walk-forward…")
def run_cv(use_alt: bool):
    market, tag = load_market()
    alt = build_alt_features(market) if use_alt else None
    X, y = build_features(market, alt=alt)
    X = X.drop(columns=X.columns[X.isna().all()])
    daily = market["spot_avg"].pct_change().reindex(X.index)
    cv = PurgedWalkForward(n_splits=6, min_train=max(400, len(X) // 4))
    out = {}
    for name, factory in MODELS.items():
        proba = pd.Series(np.nan, index=X.index)
        for tr, te in cv.split(len(X)):
            model = factory()
            arr = name == "gbm"
            model.fit(X.iloc[tr].to_numpy() if arr else X.iloc[tr], y.iloc[tr])
            proba.iloc[te] = model.predict_proba(
                X.iloc[te].to_numpy() if arr else X.iloc[te]
            )[:, 1]
        mask = proba.notna()
        bt = run_backtest(proba[mask], daily[mask], BacktestConfig())
        m = classification_metrics(y[mask], proba[mask])
        m.update(backtest_metrics(bt["net_ret"]))
        out[name] = {"metrics": m, "bt": bt, "proba": proba[mask], "y": y[mask]}
    return out, X, market, tag


# ---------------------------------------------------------------------- page
market, tag = load_market()
use_alt = st.sidebar.toggle("Alt-data features", value=True)
model_pick = st.sidebar.selectbox("Model", ["logistic", "gbm", "seasonal_baseline"])
results, X, market, tag = run_cv(use_alt)
res = results[model_pick]

banner = st.error if tag == "SYNTHETIC" else st.success
banner(
    f"**[{tag}]** "
    + (
        "Calibrated synthetic testbed — methodology demo, NOT a market finding. "
        "Run `scripts/build_dataset.py --refresh` to switch to real data."
        if tag == "SYNTHETIC"
        else f"Live dataset: {market.index.min():%d %b %Y} → {market.index.max():%d %b %Y}."
    )
)

st.title("Cardamom Quant")
st.caption(
    "Price-direction model for Indian small cardamom — leakage-safe features, "
    "purged walk-forward, honest baseline, costs included. "
    "Thesis: monsoon anomalies + rotating demand calendars + auction microstructure."
)

m = res["metrics"]
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Hit vs base", f"{m['hit_vs_base_pts']:+.1f} pts")
c2.metric("AUC", f"{m['auc']:.3f}")
c3.metric("Sharpe (net)", f"{m['sharpe']:.2f}")
c4.metric("Max drawdown", f"{m['max_dd']:.0%}")
c5.metric("Total return", f"{m['total_ret']:.0%}")

tab_px, tab_eq, tab_score, tab_cal, tab_reg, tab_cap = st.tabs(
    ["Price & signal", "Equity", "Scorecard", "Calibration", "Regimes", "Capacity"]
)

with tab_px:
    st.line_chart(market["spot_avg"].rename(f"spot ₹/kg [{tag}]"))
    st.line_chart(res["bt"]["position"].rename("position (vol-targeted weight)"))
    st.bar_chart(market["qty_arrived"].tail(120).rename("auction arrivals, kg (last ~6m)"))

with tab_eq:
    eq = pd.DataFrame(
        {name: r["bt"]["equity"] for name, r in results.items()}
    )
    st.line_chart(eq)
    st.caption("Equity, net of 15bps turnover costs, next-day execution, all models.")

with tab_score:
    rows = {n: r["metrics"] for n, r in results.items()}
    st.dataframe(
        pd.DataFrame(rows).T[
            ["hit_rate", "hit_vs_base_pts", "auc", "sharpe", "max_dd", "total_ret"]
        ].round(3)
    )
    st.caption("Out-of-sample, purged walk-forward. The seasonal baseline is the bar to beat.")

with tab_cal:
    tab = calibration_table(res["y"], res["proba"])
    st.dataframe(tab.round(3))
    chart = tab.reset_index(drop=True)[["p_mean", "y_rate"]]
    st.line_chart(chart.set_index("p_mean"))
    st.caption("Perfect calibration = the line y_rate == p_mean.")

with tab_reg:
    regs = pd.DataFrame(index=res["bt"].index)
    regs["season"] = np.where(
        pd.DatetimeIndex(regs.index).month.isin([3, 4, 5, 6, 7]), "lean", "harvest"
    )
    if "oni" in X.columns and X["oni"].notna().any():
        from src.analysis.calibration import enso_phase

        regs["enso"] = enso_phase(X["oni"].reindex(regs.index))
    st.dataframe(
        regime_performance(
            res["bt"]["net_ret"], res["y"], res["proba"], regs
        ).round(3)
    )
    st.caption("A signal that only works in one regime is a different product. Know which.")

with tab_cap:
    st.text(capacity_report(res["bt"]["position"], market))
    st.caption(
        "Participation of turnover in auction daily value. Cardamom is niche — "
        "the honest capacity number is part of the pitch, not a footnote."
    )
