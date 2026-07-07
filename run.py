"""cardamom-quant — end-to-end run.

    python run.py                 # real data if data/processed exists, else [SYNTHETIC]
    python run.py --synthetic     # force the synthetic methodology testbed
    python run.py --no-alt        # core features only (ablation)

Pipeline: dataset → features (+ alt-data block) → purged walk-forward →
out-of-sample scorecard vs the seasonal baseline → vol-targeted backtest
after costs → permutation importance of the best model.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.backtest.engine import BacktestConfig, run_backtest  # noqa: E402
from src.data import config  # noqa: E402
from src.features.alt_features import build_alt_features  # noqa: E402
from src.features.engineering import build_features, forward_returns  # noqa: E402
from src.metrics import backtest_metrics, classification_metrics, format_scorecard  # noqa: E402
from src.models.baselines import MODELS  # noqa: E402
from src.validation.walkforward import PurgedWalkForward  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("run")


def load_dataset(force_synthetic: bool) -> tuple[pd.DataFrame, str]:
    real = config.PROCESSED_DIR / "market.parquet"
    if not force_synthetic and real.exists():
        return pd.read_parquet(real), "REAL"
    from src.data.synthetic import generate_market

    log.warning("running on the SYNTHETIC testbed — results are NOT a market finding")
    return generate_market(), "SYNTHETIC"


def main(force_synthetic: bool, use_alt: bool) -> None:
    market, tag = load_dataset(force_synthetic)

    oni = None
    try:
        from src.data.climate_indices import parse_oni  # noqa: PLC0415

        oni = parse_oni((config.RAW_DIR / "climate" / "oni.ascii.txt").read_text())
    except Exception:
        log.info("ONI file not available — climate features stay NaN")
    alt = build_alt_features(market, oni=oni) if use_alt else None
    X, y = build_features(market, alt=alt)
    dead = X.columns[X.isna().all()]
    if len(dead):
        log.info("dropping %d never-observed columns (feeds not wired): %s",
                 len(dead), ", ".join(dead))
        X = X.drop(columns=dead)
    fwd = forward_returns(market).reindex(X.index)
    daily = market["spot_avg"].pct_change().reindex(X.index)

    log.info("[%s] %d samples, %d features (%s alt block)",
             tag, len(X), X.shape[1], "with" if use_alt else "without")

    cv = PurgedWalkForward(n_splits=6, min_train=max(400, len(X) // 4))
    scorecard: dict[str, dict] = {}
    oos_proba: dict[str, pd.Series] = {}

    for name, factory in MODELS.items():
        proba = pd.Series(np.nan, index=X.index)
        for tr, te in cv.split(len(X)):
            model = factory()
            Xtr, ytr = X.iloc[tr], y.iloc[tr]
            if name == "seasonal_baseline":
                model.fit(Xtr, ytr)
            else:
                model.fit(Xtr.to_numpy() if name == "gbm" else Xtr, ytr)
            Xte = X.iloc[te]
            proba.iloc[te] = model.predict_proba(
                Xte.to_numpy() if name == "gbm" else Xte
            )[:, 1]
        mask = proba.notna()
        m = classification_metrics(y[mask], proba[mask])
        bt = run_backtest(proba[mask], daily[mask], BacktestConfig())
        m.update(backtest_metrics(bt["net_ret"]))
        scorecard[name] = m
        oos_proba[name] = proba
        log.info("%-18s hit %+0.1fpts vs base | AUC %.3f | Sharpe %.2f | MaxDD %.0f%%",
                 name, m["hit_vs_base_pts"], m["auc"], m["sharpe"], m["max_dd"] * 100)

    print(f"\n=== OUT-OF-SAMPLE SCORECARD [{tag}] (after "
          f"{BacktestConfig().cost_bps:.0f}bps costs) ===")
    print(format_scorecard(scorecard))

    # permutation importance of the best non-baseline model
    best = max(
        (k for k in scorecard if k != "seasonal_baseline"),
        key=lambda k: scorecard[k]["sharpe"] if scorecard[k]["sharpe"] == scorecard[k]["sharpe"] else -9,
    )
    print(f"\n=== PERMUTATION IMPORTANCE ({best}, last fold) [{tag}] ===")
    tr, te = list(cv.split(len(X)))[-1]
    model = MODELS[best]()
    Xtr = X.iloc[tr]
    model.fit(Xtr.to_numpy() if best == "gbm" else Xtr, y.iloc[tr])
    Xte, yte = X.iloc[te], y.iloc[te]
    base_auc = classification_metrics(
        yte, pd.Series(model.predict_proba(Xte.to_numpy() if best == "gbm" else Xte)[:, 1], index=yte.index)
    )["auc"]
    rng = np.random.default_rng(0)
    drops = {}
    for col in X.columns:
        Xp = Xte.copy()
        Xp[col] = rng.permutation(Xp[col].to_numpy())
        auc = classification_metrics(
            yte, pd.Series(model.predict_proba(Xp.to_numpy() if best == "gbm" else Xp)[:, 1], index=yte.index)
        )["auc"]
        drops[col] = base_auc - auc
    for col, d in sorted(drops.items(), key=lambda kv: -kv[1])[:12]:
        print(f"  {col:<22} dAUC {d:+.4f}")

    if tag == "SYNTHETIC":
        print("\n[SYNTHETIC] Methodology validation only — synthetic Sharpes are "
              "optimistically high and are NOT presentable as a market finding.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--no-alt", action="store_true")
    a = ap.parse_args()
    main(a.synthetic, not a.no_alt)
