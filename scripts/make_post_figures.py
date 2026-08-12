"""Post figures 2 and 3 — the two charts the write-up leans on.

    python scripts/make_post_figures.py   # -> figures/turnover_story.png
                                          #    figures/ablation.png

Both recompute from the real dataset rather than transcribing a printed table,
so a figure can never quietly disagree with `scripts/analyze.py` /
`scripts/horizon_experiment.py`. They reuse those scripts' own functions, so
there is exactly one implementation of each number.

  turnover_story.png  the autopsy: a 5-day signal traded daily burns ~15x
                      annual turnover and gives the edge back in costs.
                      Matching the trade to the label cuts turnover ~4x.
  ablation.png        the honest ablation: the carefully-built alt-data block
                      makes the model WORSE, and nothing beats the one-line
                      seasonal rule at daily rebalancing.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze import run_variant  # noqa: E402
from horizon_experiment import (  # noqa: E402
    REBALANCE, THRESHOLDS, gbm_probability_streams, grid_cell,
)
from run import load_dataset  # noqa: E402
from src.analysis.robustness import ablation_table, block_bootstrap_sharpe  # noqa: E402
from src.features.engineering import HORIZON, build_features  # noqa: E402
from src.validation.walkforward import PurgedWalkForward  # noqa: E402

logging.basicConfig(level=logging.WARNING)

# palette: validated reference instance — same tokens as make_flagship_chart.py.
# Series slots 1-2 pass all-pairs CVD + normal-vision floors on this surface
# (validated: worst pair dE 24.7 protan / 33.6 normal).
SURFACE = "#fcfcfb"
SERIES_1 = "#2a78d6"   # blue   — daily rebalance
SERIES_2 = "#eb6834"   # orange — horizon-matched 5d
NEGATIVE = "#e34948"   # diverging red pole, for Sharpes below zero
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"


def _frame(ax) -> None:
    """Recessive chrome: hairline grid, no box, muted ticks."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelsize=10)
    ax.set_axisbelow(True)


def build_grid(market: pd.DataFrame) -> pd.DataFrame:
    """The 16-cell horizon x threshold grid from horizon_experiment.main()."""
    X, y = build_features(market, alt=None)
    X = X.drop(columns=X.columns[X.isna().all()])
    daily = market["spot_avg"].pct_change().reindex(X.index)
    cv = PurgedWalkForward(n_splits=6, min_train=max(400, len(X) // 4))
    streams = gbm_probability_streams(X, y, cv)

    rows = []
    for stream, proba in streams.items():
        mask = proba.notna()
        for reb in REBALANCE:
            for thr in THRESHOLDS:
                cell = grid_cell(proba[mask], daily[mask], reb, thr)
                cell.update({"stream": stream, "rebalance": reb, "thr": thr})
                rows.append(cell)
    return pd.DataFrame(rows)


def figure_turnover(grid: pd.DataFrame, baseline_sharpe: float) -> Path:
    """Paired dumbbell: each configuration traded daily vs at its own horizon.

    A scatter of Sharpe against turnover was the first attempt and it overclaimed
    — across the whole grid Sharpe is flat and the CIs swamp everything. Pairing
    each config with itself isolates the one variable that does move: the same
    signal, traded 2-3x less often, for the same Sharpe.
    """
    pairs = grid.pivot_table(index=["stream", "thr"], columns="rebalance",
                             values=["ann_turnover", "sharpe"])
    pairs = pairs.sort_values(("ann_turnover", 1))
    ratio = (pairs[("ann_turnover", 1)] / pairs[("ann_turnover", HORIZON)]).mean()
    d_sharpe = (pairs[("sharpe", HORIZON)] - pairs[("sharpe", 1)]).mean()

    fig, ax = plt.subplots(figsize=(11.5, 6.4), dpi=150)
    fig.set_facecolor(SURFACE)
    _frame(ax)

    ypos = range(len(pairs))
    for y, (_, row) in zip(ypos, pairs.iterrows()):
        t1, t5 = row[("ann_turnover", 1)], row[("ann_turnover", HORIZON)]
        ax.plot([t5, t1], [y, y], color=BASELINE, lw=1.6, zorder=2,
                solid_capstyle="round")
        ax.plot(t1, y, "o", ms=10, color=SERIES_1, markeredgecolor=SURFACE,
                markeredgewidth=2, zorder=4)
        ax.plot(t5, y, "o", ms=10, color=SERIES_2, markeredgecolor=SURFACE,
                markeredgewidth=2, zorder=4)
        ax.text(t1 + 0.45, y,
                f"Sharpe {row[('sharpe', 1)]:+.2f} → {row[('sharpe', HORIZON)]:+.2f}",
                va="center", color=INK_2, fontsize=10)

    ax.plot([], [], "o", ms=10, color=SERIES_1, label="traded daily (1d)")
    ax.plot([], [], "o", ms=10, color=SERIES_2,
            label=f"horizon-matched ({HORIZON}d)")

    ax.set_yticks(list(ypos))
    ax.set_yticklabels([f"{s} · thr {t:.2f}" for s, t in pairs.index], fontsize=10.5)
    for label in ax.get_yticklabels():
        label.set_color(INK)
    ax.set_xlim(0, pairs[("ann_turnover", 1)].max() * 1.42)

    ax.set_title(
        f"The same signal, traded {ratio:.1f}x less often — for the same Sharpe",
        color=INK, fontsize=15, fontweight="semibold", loc="left", pad=42)
    ax.text(0, 1.02,
            f"All {len(pairs)} configurations, after 15bps costs. Mean Sharpe change "
            f"from rebalancing at the label's own {HORIZON}-day horizon: "
            f"{d_sharpe:+.2f} — i.e. nothing.",
            transform=ax.transAxes, color=INK_2, fontsize=10)
    ax.set_xlabel("annualised turnover (x/yr)", color=INK_2, fontsize=11)
    ax.grid(axis="x", color=GRID, lw=1)
    leg = ax.legend(frameon=False, fontsize=10.5, loc="lower right")
    for text in leg.get_texts():
        text.set_color(INK_2)

    out = ROOT / "figures" / "turnover_story.png"
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out


def figure_ablation(table: pd.DataFrame) -> Path:
    # The seasonal baseline ignores the feature matrix, so core and core+alt
    # produce the identical number — showing it twice invites the reader to
    # think two different things were measured.
    table = table[table["variant"] != "core+alt/seasonal_baseline"].copy()
    table["variant"] = table["variant"].replace(
        {"core/seasonal_baseline": "seasonal baseline (no ML)"})
    table = table.sort_values("sharpe")
    fig, ax = plt.subplots(figsize=(11.5, 6.0), dpi=150)
    fig.set_facecolor(SURFACE)
    _frame(ax)

    colors = [SERIES_1 if v >= 0 else NEGATIVE for v in table["sharpe"]]
    bars = ax.barh(table["variant"], table["sharpe"], color=colors, height=0.62,
                   zorder=3)
    ax.bar_label(bars, labels=[f"{v:+.2f}" for v in table["sharpe"]],
                 padding=6, color=INK_2, fontsize=10.5, fontweight="semibold")

    base = table.loc[table["variant"] == "seasonal baseline (no ML)", "sharpe"]
    if len(base):
        ax.axvline(float(base.iloc[0]), color=MUTED, lw=1, ls=(0, (5, 4)), zorder=2)
    ax.axvline(0, color=BASELINE, lw=1.2, zorder=2)

    ax.set_title("Adding the alt-data block made the model worse",
                 color=INK, fontsize=15, fontweight="semibold", loc="left", pad=42)
    ax.text(0, 1.02,
            "Purged walk-forward Sharpe after 15bps costs, daily rebalancing. "
            "'+alt' adds rain, ENSO/IOD, calendars, microstructure and Guatemala supply.",
            transform=ax.transAxes, color=INK_2, fontsize=10)
    ax.set_xlabel("Sharpe ratio, net of costs", color=INK_2, fontsize=11)
    ax.tick_params(axis="y", labelsize=11)
    for label in ax.get_yticklabels():
        label.set_color(INK)
    ax.grid(axis="x", color=GRID, lw=1)
    ax.margins(x=0.16)

    out = ROOT / "figures" / "ablation.png"
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    market, tag = load_dataset(False)
    if tag != "REAL":
        raise SystemExit(f"refusing to publish figures built from {tag} data")

    results = {}
    for variant, use_alt in (("core", False), ("core+alt", True)):
        results[variant], _ = run_variant(market, use_alt)
    rows = {}
    for variant, models in results.items():
        for name, r in models.items():
            m = dict(r["metrics"])
            if name != "seasonal_baseline":
                m.update(block_bootstrap_sharpe(r["net_ret"]))
            rows[f"{variant}/{name}"] = m
    table = ablation_table(rows)          # indexed by variant
    if "variant" not in table.columns:    # keep it as a column for plotting
        table = table.reset_index()
    baseline_sharpe = float(
        table.loc[table["variant"] == "core/seasonal_baseline", "sharpe"].iloc[0]
    )

    grid = build_grid(market)

    for path in (figure_turnover(grid, baseline_sharpe), figure_ablation(table)):
        print(f"saved {path}")


if __name__ == "__main__":
    main()
