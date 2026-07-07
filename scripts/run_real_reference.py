from __future__ import annotations
import argparse, logging, sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.backtest.engine import BacktestConfig, run_backtest
from src.data import config
from src.features.alt_features import build_alt_features
from src.features.engineering import build_features
from src.metrics import backtest_metrics, classification_metrics, format_scorecard
from src.models.baselines import MODELS
from src.validation.walkforward import PurgedWalkForward
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("run")

def load_dataset(force_synthetic: bool):
    real = config.PROCESSED_DIR / "market.parquet"
    if not force_synthetic and real.exists():
        return pd.read_parquet(real), "REAL"
    from src.data.synthetic import generate_market
    return generate_market(), "SYNTHETIC"

def main(force_synthetic, use_alt):
    market, tag = load_dataset(force_synthetic)
    oni = None
    try:
        from src.data.climate_indices import parse_oni
        oni = parse_oni((config.RAW_DIR / "climate" / "oni.ascii.txt").read_text())
    except Exception:
        pass
    alt = build_alt_features(market, oni=oni) if use_alt else None
    X, y = build_features(market, alt=alt)
    dead = X.columns[X.isna().all()]
    if len(dead):
        log.info("dropping never-observed cols: %s", ", ".join(dead))
        X = X.drop(columns=dead)
    daily = market["spot_avg"].pct_change().reindex(X.index)
    log.info("[%s] %d samples, %d features", tag, len(X), X.shape[1])
    cv = PurgedWalkForward(n_splits=6, min_train=max(400, len(X)//4))
    scorecard = {}
    for name, factory in MODELS.items():
        proba = pd.Series(np.nan, index=X.index)
        for tr, te in cv.split(len(X)):
            model = factory()
            arr = name == "gbm"
            model.fit(X.iloc[tr].to_numpy() if arr else X.iloc[tr], y.iloc[tr])
            proba.iloc[te] = model.predict_proba(X.iloc[te].to_numpy() if arr else X.iloc[te])[:, 1]
        mask = proba.notna()
        m = classification_metrics(y[mask], proba[mask])
        bt = run_backtest(proba[mask], daily[mask], BacktestConfig())
        m.update(backtest_metrics(bt["net_ret"]))
        scorecard[name] = m
    print(f"\n=== OUT-OF-SAMPLE SCORECARD [{tag}] (after 15bps costs) ===")
    print(format_scorecard(scorecard))
    best = max((k for k in scorecard if k != "seasonal_baseline"),
               key=lambda k: scorecard[k]["sharpe"] if scorecard[k]["sharpe"]==scorecard[k]["sharpe"] else -9)
    print(f"\n=== PERMUTATION IMPORTANCE ({best}, last fold) [{tag}] ===")
    tr, te = list(cv.split(len(X)))[-1]
    model = MODELS[best]()
    model.fit(X.iloc[tr].to_numpy() if best=="gbm" else X.iloc[tr], y.iloc[tr])
    Xte, yte = X.iloc[te], y.iloc[te]
    base_auc = classification_metrics(yte, pd.Series(model.predict_proba(Xte.to_numpy() if best=="gbm" else Xte)[:,1], index=yte.index))["auc"]
    rng = np.random.default_rng(0)
    drops = {}
    for col in X.columns:
        Xp = Xte.copy(); Xp[col] = rng.permutation(Xp[col].to_numpy())
        auc = classification_metrics(yte, pd.Series(model.predict_proba(Xp.to_numpy() if best=="gbm" else Xp)[:,1], index=yte.index))["auc"]
        drops[col] = base_auc - auc
    for col, d in sorted(drops.items(), key=lambda kv: -kv[1])[:12]:
        print(f"  {col:<22} dAUC {d:+.4f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--no-alt", action="store_true")
    a = ap.parse_args()
    main(a.synthetic, not a.no_alt)
