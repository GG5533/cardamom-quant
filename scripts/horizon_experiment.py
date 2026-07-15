"""Horizon-matched trading experiment — the make-or-break test.

    python scripts/horizon_experiment.py
    python scripts/horizon_experiment.py --synthetic

RESULTS_REAL.md's finding: the 5d-label GBM signal is real (+4.2pts vs
base) but daily rebalancing burns it in turnover. This script asks the one
question that verdict leaves open: does trading at the label's own horizon
(and/or only on conviction) recover the edge after costs?

Design:
  * candidate: core-features GBM (alt features hurt on real data), two
    probability streams — raw, and per-fold isotonic-calibrated on a
    purged tail slice of each training window (probabilities feed 2p-1
    sizing, so calibration is a sizing fix, not cosmetics);
  * grid: rebalance ∈ {1d, 5d} × conviction threshold ∈ {0, .05, .10, .15}
    — 16 cells, each a fully-counted trial;
  * verdict standard: 90% block-bootstrap CI on the annualized Sharpe must
    clear zero, and the result must survive the deflated-Sharpe haircut
    over EVERY trial ever run on this data (6 prior + 16 grid cells + 2
    tranched books = 24);
  * anchor honesty: a 5d cadence hides an anchor-day choice, so the
    deliverable is the anchor-free tranched book (1/5 of capital
    rebalances daily), reported with the full anchor spread.
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
sys.path.insert(0, str(ROOT / "scripts"))

from analyze import run_variant  # noqa: E402  (the 6 prior-trial Sharpes)
from run import load_dataset  # noqa: E402
from src.analysis.calibration import calibration_summary, isotonic_calibrator  # noqa: E402
from src.analysis.robustness import block_bootstrap_sharpe, deflated_sharpe  # noqa: E402
from src.backtest.engine import BacktestConfig, run_backtest  # noqa: E402
from src.features.engineering import HORIZON, build_features  # noqa: E402
from src.metrics import backtest_metrics  # noqa: E402
from src.models.baselines import make_gbm  # noqa: E402
from src.validation.walkforward import PurgedWalkForward  # noqa: E402

logging.basicConfig(level=logging.WARNING)

THRESHOLDS = (0.0, 0.05, 0.10, 0.15)
REBALANCE = (1, HORIZON)
CALIB_FRACTION = 0.2  # tail share of each training window used for isotonic


def gbm_probability_streams(
    X: pd.DataFrame, y: pd.Series, cv: PurgedWalkForward
) -> dict[str, pd.Series]:
    """OOS P(up) per fold: raw (full-train fit, the published pipeline) and
    calibrated (fit on the front of the train window, isotonic on a purged
    tail slice, both applied to the untouched test block)."""
    raw = pd.Series(np.nan, index=X.index)
    cal = pd.Series(np.nan, index=X.index)
    for tr, te in cv.split(len(X)):
        Xte = X.iloc[te].to_numpy()

        model = make_gbm()
        model.fit(X.iloc[tr].to_numpy(), y.iloc[tr])
        raw.iloc[te] = model.predict_proba(Xte)[:, 1]

        n_cal = max(50, int(len(tr) * CALIB_FRACTION))
        inner = tr[: -(n_cal + HORIZON)]          # purge before the slice
        calib = tr[-n_cal:]
        base = make_gbm()
        base.fit(X.iloc[inner].to_numpy(), y.iloc[inner])
        iso = isotonic_calibrator(
            pd.Series(base.predict_proba(X.iloc[calib].to_numpy())[:, 1]),
            pd.Series(y.iloc[calib].to_numpy()),
        )
        cal.iloc[te] = iso.predict(base.predict_proba(Xte)[:, 1])
    return {"raw": raw, "calibrated": cal}


def tranched_net_returns(
    proba: pd.Series, daily: pd.Series, reb: int = HORIZON
) -> tuple[pd.Series, list[float]]:
    """Staggered implementation: 1/reb of the book rebalances each day.

    A single 5d-rebalance backtest secretly picks an anchor day; the
    Jegadeesh-Titman overlapping-portfolio fix trades `reb` books offset by
    one day each and averages them. Costs are paid per book (no internal
    netting), so the aggregate is conservative. Returns the averaged net
    return stream plus each anchor's standalone Sharpe (the honesty spread).
    """
    nets, anchor_sharpes = [], []
    for k in range(reb):
        bt = run_backtest(
            proba.iloc[k:], daily.iloc[k:], BacktestConfig(rebalance_every=reb)
        )
        nets.append(bt["net_ret"])
        anchor_sharpes.append(backtest_metrics(bt["net_ret"])["sharpe"])
    combined = pd.concat(nets, axis=1).mean(axis=1).dropna()
    return combined, anchor_sharpes


def grid_cell(proba: pd.Series, daily: pd.Series, reb: int, thr: float) -> dict:
    bt = run_backtest(
        proba, daily,
        BacktestConfig(rebalance_every=reb, conviction_threshold=thr),
    )
    m = backtest_metrics(bt["net_ret"])
    m["ann_turnover"] = float(bt["turnover"].mean() * 252)
    m["days_in_mkt"] = float((bt["position"].abs() > 1e-9).mean())
    net = bt["net_ret"].dropna()
    if len(net) > 63 and net.std() > 0:
        m.update(block_bootstrap_sharpe(bt["net_ret"]))
    else:  # never trades (threshold too high) — degenerate cell
        m.update({"ci_5": np.nan, "ci_95": np.nan, "p_leq_0": np.nan})
    return m


def main(force_synthetic: bool) -> None:
    market, tag = load_dataset(force_synthetic)
    X, y = build_features(market, alt=None)
    X = X.drop(columns=X.columns[X.isna().all()])
    daily = market["spot_avg"].pct_change().reindex(X.index)
    cv = PurgedWalkForward(n_splits=6, min_train=max(400, len(X) // 4))

    # ---- prior trials (the standing 6-entry ledger from analyze.py) --------
    prior_sharpes, baseline_sharpe = [], np.nan
    for use_alt in (False, True):
        variant, _ = run_variant(market, use_alt)
        if not use_alt:
            baseline_sharpe = variant["seasonal_baseline"]["metrics"]["sharpe"]
        prior_sharpes += [
            r["metrics"]["sharpe"] for r in variant.values()
            if r["metrics"]["sharpe"] == r["metrics"]["sharpe"]
        ]

    # ---- probability streams ----------------------------------------------
    streams = gbm_probability_streams(X, y, cv)
    mask = streams["raw"].notna()
    print(f"\n=== CALIBRATION CHECK (core/gbm, OOS) [{tag}] ===")
    for name, proba in streams.items():
        c = calibration_summary(y[mask], proba[mask])
        print(f"  {name:<11} Brier {c['brier']:.4f} vs climatology "
              f"{c['brier_climatology']:.4f} | max bin gap {c['max_abs_gap']:+.3f}")

    # ---- the grid -----------------------------------------------------------
    rows, ledger = [], list(prior_sharpes)
    for name, proba in streams.items():
        for reb in REBALANCE:
            for thr in THRESHOLDS:
                m = grid_cell(proba[mask], daily[mask], reb, thr)
                if m["sharpe"] == m["sharpe"]:
                    ledger.append(m["sharpe"])
                rows.append(
                    {
                        "stream": name, "rebalance": f"{reb}d", "thr": thr,
                        "sharpe": m["sharpe"], "ci_5": m.get("ci_5"),
                        "ci_95": m.get("ci_95"), "p_leq_0": m.get("p_leq_0"),
                        "max_dd": m["max_dd"], "total_ret": m["total_ret"],
                        "ann_turnover": m["ann_turnover"],
                        "days_in_mkt": m["days_in_mkt"],
                    }
                )
    grid = pd.DataFrame(rows).set_index(["stream", "rebalance", "thr"])
    print(f"\n=== HORIZON-MATCHED TRADING GRID (core/gbm) [{tag}] ===")
    print(f"    seasonal-baseline yardstick: Sharpe {baseline_sharpe:+.2f} (daily rebalance)")
    print(grid.round(3).to_string())

    clears = grid[(grid["ci_5"] > 0)]
    if len(clears):
        print(f"\n  {len(clears)} cell(s) whose 90% CI clears zero:")
        print(clears.round(3).to_string())
    else:
        print("\n  No cell's 90% CI clears zero — the honest verdict stands.")

    # ---- headline: anchor-free tranched implementation ---------------------
    # A single 5d cell picks an arbitrary anchor day; the deliverable is the
    # staggered book (thr=0 fixed ex ante — the gate never helped at 5d).
    print(f"\n=== TRANCHED {HORIZON}d IMPLEMENTATION (anchor-free) [{tag}] ===")
    for name, proba in streams.items():
        net, anchors = tranched_net_returns(proba[mask], daily[mask])
        m = backtest_metrics(net)
        ledger.append(m["sharpe"])  # the tranched books are trials too
        boot = block_bootstrap_sharpe(net)
        dsr = deflated_sharpe(net, ledger)
        a = np.array(anchors)
        print(f"  {name}")
        print(f"    anchor Sharpes        : {np.round(a, 2)} "
              f"(mean {a.mean():+.2f}, min {a.min():+.2f})")
        print(f"    tranched Sharpe       : {m['sharpe']:+.2f}  90% CI "
              f"[{boot['ci_5']:+.2f}, {boot['ci_95']:+.2f}]  "
              f"p(SR<=0) {boot['p_leq_0']:.3f}")
        print(f"    max_dd {m['max_dd']:+.1%}  total_ret {m['total_ret']:+.1%}")
        print(f"    DSR over {dsr['n_trials']} trials  : {dsr['dsr']:.3f} "
              f"(expected max SR from luck {dsr['expected_max_sharpe']:.2f})")
    if tag == "SYNTHETIC":
        print("\n[SYNTHETIC] methodology validation only — not a market finding.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    main(ap.parse_args().synthetic)
