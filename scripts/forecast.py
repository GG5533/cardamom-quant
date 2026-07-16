"""Live champion forecast + prospective scoring — run weekly, after
scripts/refresh_spot.py.

    python scripts/forecast.py

What it does, in order:
  1. verifies both ledger hash chains (any past edit aborts everything);
  2. rebuilds the champion (pre-rain core + auction physics, calibrated
     GBM) on ALL labeled auction days;
  3. emits P(up, 5d) for the current unlabeled tail days — forecasts
     written BEFORE their outcomes exist — into the append-only,
     hash-chained forecast ledger;
  4. scores every previously-logged forecast whose 5-day outcome has now
     matured, into the outcome ledger;
  5. prints the running prospective scorecard.

Feature note: build_features() drops label-undefined tail rows, which are
precisely the rows a live forecast is about. We therefore extend the
market frame with HORIZON synthetic future rows before building features.
Every feature is strictly backward-looking (shift(1) discipline), so the
synthetic rows cannot contaminate real rows' features — asserted by
tests/test_live_ledger.py::test_tail_features_match_unextended.
"""
from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features.auction_physics import build_physics_features  # noqa: E402
from src.features.engineering import HORIZON, build_features  # noqa: E402
from src.live.ledger import forecast_ledger, outcome_ledger  # noqa: E402
from src.models.baselines import make_gbm  # noqa: E402

from sklearn.isotonic import IsotonicRegression  # noqa: E402

MODEL_TAG = "champion-physics-gbm"
CALIB_FRACTION = 0.2


def features_all_rows(market: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Champion feature matrix for EVERY real auction day, tail included.

    Returns (X, labeled) where labeled marks rows whose 5d label is real.
    """
    last = market.index[-1]
    synth_idx = pd.bdate_range(last, periods=HORIZON + 1, freq="B")[1:]
    synth = market.reindex(market.index.union(synth_idx)).ffill()
    X, _ = build_features(synth, alt=build_physics_features(synth))
    X = X.loc[X.index.isin(market.index)]
    X = X.drop(columns=X.columns[X.isna().all()])
    labeled = pd.Series(
        X.index <= market.index[-(HORIZON + 1)], index=X.index
    )
    return X, labeled


def fit_champion(X: pd.DataFrame, y: pd.Series):
    """Full-data champion fit: GBM on the front, isotonic on a purged tail."""
    n = len(X)
    n_cal = max(50, int(n * CALIB_FRACTION))
    inner = slice(0, n - n_cal - HORIZON)
    calib = slice(n - n_cal, n)
    base = make_gbm()
    base.fit(X.iloc[inner].to_numpy(), y.iloc[inner])
    iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
    iso.fit(base.predict_proba(X.iloc[calib].to_numpy())[:, 1],
            y.iloc[calib].to_numpy())
    return lambda A: iso.predict(base.predict_proba(A)[:, 1])


def main() -> None:
    fc, oc = forecast_ledger(), outcome_ledger()
    print(f"ledger integrity: forecasts {fc.verify_chain()} rows OK, "
          f"outcomes {oc.verify_chain()} rows OK")

    market = pd.read_parquet(ROOT / "data" / "processed" / "market.parquet")
    market = market.drop(columns=["rain_mm", "rain_climatology", "rain_anomaly"])
    px = market["spot_avg"]
    X, labeled = features_all_rows(market)

    # forward labels for the labeled region (same construction as run.py)
    fwd = (px.shift(-HORIZON) / px - 1.0).reindex(X.index)
    y = (fwd > 0).astype(float)

    predict = fit_champion(X[labeled], y[labeled])

    # ---- emit forecasts for unlabeled tail days not yet in the ledger -----
    have = set(fc.read()["auction_date"])
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                         cwd=ROOT, capture_output=True, text=True).stdout.strip()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    new = 0
    for d in X.index[~labeled]:
        ds = d.strftime("%Y-%m-%d")
        if ds in have:
            continue
        p = float(predict(X.loc[[d]].to_numpy())[0])
        fc.append({
            "auction_date": ds, "made_on_utc": now,
            "p_up": f"{p:.4f}", "signal": f"{2 * p - 1:+.4f}",
            "spot_avg": f"{px.loc[d]:.2f}", "model": MODEL_TAG, "git_sha": sha,
        })
        new += 1
        print(f"forecast {ds}: P(up,5d) = {p:.3f}  signal {2*p-1:+.3f}")
    print(f"{new} new forecast(s) logged")

    # ---- score matured forecasts -------------------------------------------
    fdf = fc.read()
    scored = set(oc.read()["auction_date"])
    pos = {d.strftime("%Y-%m-%d"): i for i, d in enumerate(px.index)}
    n_scored = 0
    for _, row in fdf.iterrows():
        ds = row["auction_date"]
        if ds in scored or ds not in pos:
            continue
        i = pos[ds]
        if i + HORIZON >= len(px):
            continue  # not matured yet
        ret = float(px.iloc[i + HORIZON] / px.iloc[i] - 1.0)
        up = float(ret > 0)
        p = float(row["p_up"])
        oc.append({
            "auction_date": ds,
            "scored_on_utc": now,
            "fwd_5d_ret": f"{ret:+.5f}",
            "outcome_up": f"{up:.0f}",
            "hit": f"{float((p > 0.5) == (up > 0.5)):.0f}",
            "brier": f"{(p - up) ** 2:.4f}",
        })
        n_scored += 1

    odf = oc.read()
    print(f"{n_scored} forecast(s) matured and scored")
    if len(odf):
        hit = odf["hit"].astype(float).mean()
        brier = odf["brier"].astype(float).mean()
        print(f"\n=== PROSPECTIVE SCORECARD (all-time) ===")
        print(f"  scored forecasts : {len(odf)}")
        print(f"  hit rate         : {hit:.1%}")
        print(f"  Brier            : {brier:.4f} (climatology ~0.25)")
    else:
        print("no matured outcomes yet — the clock is running")


if __name__ == "__main__":
    main()
