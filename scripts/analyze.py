"""Full research report: ablation, robustness, calibration, regimes, SHAP.

    python scripts/analyze.py               # real data if built, else [SYNTHETIC]
    python scripts/analyze.py --synthetic

Runs the same purged walk-forward as run.py across TWO feature variants
(core, core+alt) x three models, then asks the questions a desk risk
committee would ask:
  - is the Sharpe distinguishable from zero? (block bootstrap)
  - does it survive a multiple-testing haircut? (deflated Sharpe over ALL
    six trials — every model/variant evaluated counts as a trial)
  - do the probabilities mean anything? (calibration, Brier vs climatology)
  - when does it work? (ENSO phase / crop season splits)
  - why does it work? (SHAP on the best model)
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run import load_dataset  # noqa: E402  (reuses run.py's data loading)
from src.analysis.calibration import (  # noqa: E402
    calibration_summary, calibration_table, enso_phase, regime_performance,
)
from src.analysis.interpret import shap_report  # noqa: E402
from src.analysis.robustness import (  # noqa: E402
    ablation_table, block_bootstrap_sharpe, deflated_sharpe,
)
from src.backtest.engine import BacktestConfig, run_backtest  # noqa: E402
from src.features.alt_features import build_alt_features  # noqa: E402
from src.features.engineering import build_features  # noqa: E402
from src.metrics import backtest_metrics, classification_metrics  # noqa: E402
from src.models.baselines import MODELS  # noqa: E402
from src.validation.walkforward import PurgedWalkForward  # noqa: E402

logging.basicConfig(level=logging.WARNING)


def run_variant(market: pd.DataFrame, use_alt: bool):
    alt = build_alt_features(market) if use_alt else None
    X, y = build_features(market, alt=alt)
    X = X.drop(columns=X.columns[X.isna().all()])
    daily = market["spot_avg"].pct_change().reindex(X.index)
    cv = PurgedWalkForward(n_splits=6, min_train=max(400, len(X) // 4))
    out = {}
    for name, factory in MODELS.items():
        proba = pd.Series(np.nan, index=X.index)
        last_fit = None
        for tr, te in cv.split(len(X)):
            model = factory()
            arr = name == "gbm"
            model.fit(X.iloc[tr].to_numpy() if arr else X.iloc[tr], y.iloc[tr])
            proba.iloc[te] = model.predict_proba(
                X.iloc[te].to_numpy() if arr else X.iloc[te]
            )[:, 1]
            last_fit, last_te = model, te
        mask = proba.notna()
        bt = run_backtest(proba[mask], daily[mask], BacktestConfig())
        m = classification_metrics(y[mask], proba[mask])
        m.update(backtest_metrics(bt["net_ret"]))
        out[name] = {
            "metrics": m, "proba": proba[mask], "y": y[mask],
            "net_ret": bt["net_ret"], "model": last_fit,
            "X_last_te": X.iloc[last_te],
        }
    return out, X


def main(force_synthetic: bool) -> None:
    market, tag = load_dataset(force_synthetic)
    results = {}
    for variant, use_alt in (("core", False), ("core+alt", True)):
        results[variant], X_full = run_variant(market, use_alt)

    # ---------------- ablation + robustness --------------------------------
    all_trial_sharpes = [
        r["metrics"]["sharpe"]
        for v in results.values() for r in v.values()
        if r["metrics"]["sharpe"] == r["metrics"]["sharpe"]
    ]
    rows = {}
    for variant, models in results.items():
        for name, r in models.items():
            key = f"{variant}/{name}"
            m = dict(r["metrics"])
            if name != "seasonal_baseline":
                m.update(block_bootstrap_sharpe(r["net_ret"]))
            rows[key] = m
    print(f"\n=== ABLATION x ROBUSTNESS [{tag}] ===")
    print(ablation_table(rows).round(3).to_string())

    # best non-baseline trial by Sharpe
    best_key = max(
        (k for k in rows if "baseline" not in k),
        key=lambda k: rows[k].get("sharpe", -9) if rows[k].get("sharpe") == rows[k].get("sharpe") else -9,
    )
    variant, best_name = best_key.split("/")
    best = results[variant][best_name]

    dsr = deflated_sharpe(best["net_ret"], all_trial_sharpes)
    print(f"\n=== DEFLATED SHARPE ({best_key}) [{tag}] ===")
    print(f"  trials counted        : {dsr['n_trials']}")
    print(f"  expected max SR (luck): {dsr['expected_max_sharpe']:.2f}")
    print(f"  PSR vs 0              : {dsr['psr_vs_zero']:.3f}")
    print(f"  DSR (survives search?): {dsr['dsr']:.3f}")

    # ---------------- calibration ------------------------------------------
    cal = calibration_summary(best["y"], best["proba"])
    print(f"\n=== CALIBRATION ({best_key}) [{tag}] ===")
    print(f"  Brier {cal['brier']:.4f} vs climatology {cal['brier_climatology']:.4f} "
          f"| max bin gap {cal['max_abs_gap']:+.3f}")
    print(calibration_table(best["y"], best["proba"]).round(3).to_string())

    # ---------------- regimes ----------------------------------------------
    regs = pd.DataFrame(index=best["net_ret"].index)
    regs["season"] = np.where(
        pd.DatetimeIndex(regs.index).month.isin([3, 4, 5, 6, 7]), "lean", "harvest"
    )
    if "oni" in X_full.columns and X_full["oni"].notna().any():
        regs["enso"] = enso_phase(X_full["oni"].reindex(regs.index))
    print(f"\n=== REGIME PERFORMANCE ({best_key}) [{tag}] ===")
    print(regime_performance(best["net_ret"], best["y"], best["proba"], regs)
          .round(3).to_string())

    # ---------------- SHAP --------------------------------------------------
    print(f"\n=== SHAP ({best_key}, last test fold) [{tag}] ===")
    print(shap_report(best_name, best["model"], best["X_last_te"]))

    if tag == "SYNTHETIC":
        print("\n[SYNTHETIC] methodology validation only — not a market finding.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    main(ap.parse_args().synthetic)
