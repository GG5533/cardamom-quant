"""Every model in the project, evaluated in one vehicle, in one figure.

    python scripts/make_model_matrix.py            # compute + draw (~20-30 min)
    python scripts/make_model_matrix.py --redraw   # redraw from the cache only

  -> figures/model_matrix.png
     figures/model_matrix.json   (the cache; delete it to force a recompute)

Everything is recomputed from the current dataset through the SAME functions the
trial scripts use (edge_hunt.calibrated_fold_proba / evaluate_stream,
horizon_experiment.tranched_net_returns, robust_estimate.champion_estimate), so
no number here can drift from the script it illustrates.

Scope, stated because it is a real limit: this compares the designs that produce
a *probability stream* and can therefore share the tranched 5-day vehicle. Three
models in the repo cannot go on this axis honestly --

  T3 ou-bands       a weights policy, not a probability stream (own vehicle)
  T4 conformal-gbm  a gate on a regressor, evaluated differently
  T9/T10 guatemala  needs the banguat loader; the columns are not in the parquet

-- and one, T14 mcx-basis, cannot run at all right now: the weekly parquet
rebuild drops basis_pct, so merge_mcx.py has to be re-run first.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze import run_variant  # noqa: E402
from edge_hunt import calibrated_fold_proba, evaluate_stream  # noqa: E402
from horizon_experiment import tranched_net_returns  # noqa: E402
from robust_estimate import LAYOUTS, champion_estimate  # noqa: E402
from run import load_dataset  # noqa: E402
from src.analysis.robustness import block_bootstrap_sharpe  # noqa: E402
from src.features.auction_physics import build_physics_features  # noqa: E402
from src.features.engineering import build_features  # noqa: E402
from src.metrics import backtest_metrics  # noqa: E402
from src.models.kalman_seasonal import kalman_features  # noqa: E402
from src.validation.walkforward import PurgedWalkForward  # noqa: E402

logging.basicConfig(level=logging.WARNING)

CACHE = ROOT / "figures" / "model_matrix.json"

# palette: validated reference instance, same tokens as the other post figures
SURFACE = "#fcfcfb"
SERIES_1 = "#2a78d6"
NEGATIVE = "#e34948"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

# Group 2 is deliberately labelled with its fold layout. Single-layout Sharpes in
# this project swing enormously -- core-gbm measures +0.84 on "6 folds +0" and
# +0.10 on "5 folds +63", and blends to +0.46. Reading group 2 as a ranking is
# the exact error the blend in group 3 exists to prevent, so the axis says so.
GROUPS = ["daily rebalance", "tranched 5-day vehicle · single 6-fold layout",
          "slicing-robust blend · all 6 layouts"]


def blend_layouts(X, y, daily) -> dict:
    """Average one design's probability stream over all six fold layouts.

    The same construction champion_estimate() applies to the physics design,
    generalised so any design can be held to the same standard.
    """
    base_min = max(400, len(y) // 4)
    streams = []
    for n_splits, off in LAYOUTS:
        cv = PurgedWalkForward(n_splits=n_splits, min_train=base_min + off)
        streams.append(calibrated_fold_proba(X, y, cv))
    blend = pd.concat(streams, axis=1).mean(axis=1)
    mask = blend.notna()
    net, _ = tranched_net_returns(blend[mask], daily[mask])
    m = backtest_metrics(net)
    m.update(block_bootstrap_sharpe(net))
    return m


def compute() -> list[dict]:
    market, tag = load_dataset(False)
    if tag != "REAL":
        raise SystemExit(f"refusing to build the matrix from {tag} data")
    rows: list[dict] = []

    # ---- group 1: the core registry, daily rebalance -----------------------
    core, _ = run_variant(market, use_alt=False)
    for name, r in core.items():
        rows.append({"group": GROUPS[0], "model": name,
                     "sharpe": r["metrics"]["sharpe"], "ci_5": None, "ci_95": None})
        print(f"  {name:26} {r['metrics']['sharpe']:+.3f}")

    # ---- group 2: probability streams in the tranched 5d vehicle -----------
    X_core, y = build_features(market, alt=None)
    X_core = X_core.drop(columns=X_core.columns[X_core.isna().all()])
    daily = market["spot_avg"].pct_change().reindex(X_core.index)
    px = market["spot_avg"].reindex(X_core.index)
    cv = PurgedWalkForward(n_splits=6, min_train=max(400, len(X_core) // 4))

    # rain lives in the parquet, so "core" above already carries it; the physics
    # designs deliberately drop it, matching how edge_hunt builds them.
    pre = market.drop(columns=["rain_mm", "rain_climatology", "rain_anomaly"])
    X_pre, y_pre = build_features(pre, alt=None)
    X_pre = X_pre.drop(columns=X_pre.columns[X_pre.isna().all()])

    X_phys, y_phys = build_features(pre, alt=build_physics_features(pre))
    X_phys = X_phys.drop(columns=X_phys.columns[X_phys.isna().all()])

    X_pr, y_pr = build_features(market, alt=build_physics_features(market))
    X_pr = X_pr.drop(columns=X_pr.columns[X_pr.isna().all()])

    def x_with_kalman(train_end: int) -> pd.DataFrame:
        return X_pre.join(kalman_features(px, train_end)[0])

    def x_phys_kalman(train_end: int) -> pd.DataFrame:
        return X_phys.join(kalman_features(px, train_end)[0])

    designs = [
        ("core-gbm (no physics)", X_pre, y_pre),
        ("T6 rain-gbm", X_core, y),
        ("T1 physics-gbm", X_phys, y_phys),
        ("T2 kalman-gbm", x_with_kalman, y_pre),
        ("T5 physics+kalman", x_phys_kalman, y_phys),
        ("T7 physics+rain-gbm", X_pr, y_pr),
    ]
    for label, X, yy in designs:
        m = evaluate_stream(label, calibrated_fold_proba(X, yy, cv), yy, daily, [])
        rows.append({"group": GROUPS[1], "model": label, "sharpe": m["sharpe"],
                     "ci_5": m.get("ci_5"), "ci_95": m.get("ci_95")})
        print(f"  {label:26} {m['sharpe']:+.3f}")

    # ---- group 3: the slicing-robust blend ---------------------------------
    # Both arms, because the single-layout group above makes core look like it
    # beats the champion (+0.84 vs +0.73) and the blend is what refutes that.
    _, champ = champion_estimate(market)
    rows.append({"group": GROUPS[2], "model": "champion: physics (blended)",
                 "sharpe": champ["sharpe"], "ci_5": champ["ci_5"],
                 "ci_95": champ["ci_95"]})
    print(f"  {'champion blended':26} {champ['sharpe']:+.3f}")

    core_b = blend_layouts(X_pre, y_pre, daily)
    rows.append({"group": GROUPS[2], "model": "core, no physics (blended)",
                 "sharpe": core_b["sharpe"], "ci_5": core_b["ci_5"],
                 "ci_95": core_b["ci_95"]})
    print(f"  {'core blended':26} {core_b['sharpe']:+.3f}")

    CACHE.write_text(json.dumps(rows, indent=2))
    return rows


def draw(rows: list[dict]) -> Path:
    fig, ax = plt.subplots(figsize=(11.5, 8.4), dpi=150)
    fig.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelsize=10)
    ax.set_axisbelow(True)

    finite = [v for r in rows for v in (r["ci_5"], r["ci_95"], r["sharpe"])
              if v is not None and v == v]
    xmin, xmax = min(min(finite), 0) - 0.18, max(max(finite), 0) + 0.14

    ypos, labels = [], []
    pos = 0
    for g in GROUPS:
        sub = sorted([r for r in rows if r["group"] == g],
                     key=lambda r: r["sharpe"], reverse=True)
        if not sub:
            continue
        ax.text(xmin + 0.015, pos - 0.85, g.upper(), color=MUTED, fontsize=9,
                fontweight="semibold", va="center")
        for r in sub:
            ypos.append(pos)
            labels.append(r["model"])
            colour = SERIES_1 if r["sharpe"] >= 0 else NEGATIVE
            if r["ci_5"] is not None and r["ci_5"] == r["ci_5"]:
                ax.plot([r["ci_5"], r["ci_95"]], [pos, pos], color=colour,
                        lw=1.8, alpha=0.5, solid_capstyle="round", zorder=3)
            ax.plot(r["sharpe"], pos, "o", ms=10, color=colour,
                    markeredgecolor=SURFACE, markeredgewidth=2, zorder=4)
            ax.text(r["sharpe"], pos - 0.36, f"{r['sharpe']:+.2f}", ha="center",
                    color=INK_2, fontsize=9.5, fontweight="semibold")
            pos += 1
        pos += 1.6                      # gap between groups

    ax.axvline(0, color=BASELINE, lw=1.4, zorder=2)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=10.5)
    for lab in ax.get_yticklabels():
        lab.set_color(INK)
    ax.set_ylim(pos - 1.2, -1.4)        # inverted: first group on top
    ax.set_xlim(xmin, xmax)

    ax.set_title("Every model in the project, one vehicle at a time",
                 color=INK, fontsize=15, fontweight="semibold", loc="left", pad=46)
    ax.text(0, 1.015,
            "Sharpe net of 15bps costs, purged walk-forward, out-of-sample. Bars are 90% "
            "block-bootstrap CIs where the vehicle produces one.\nGroups are not comparable "
            "across — different bets. Do not rank within group 2 either: those are ONE fold "
            "layout, and re-slicing moves them by\nmore than the gaps between them. Group 3 "
            "is the number that survives re-slicing.",
            transform=ax.transAxes, color=INK_2, fontsize=9.5, linespacing=1.5)
    ax.set_xlabel("Sharpe ratio, net of costs", color=INK_2, fontsize=11)
    ax.grid(axis="x", color=GRID, lw=1)

    out = ROOT / "figures" / "model_matrix.png"
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--redraw", action="store_true",
                    help="skip the compute and redraw from figures/model_matrix.json")
    args = ap.parse_args()
    data = json.loads(CACHE.read_text()) if args.redraw else compute()
    print(f"saved {draw(data)}")
