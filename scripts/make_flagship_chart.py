"""Flagship figure: 12 years of cardamom spot history, with the two events
that shaped this project — the 2018-flood price spike (the pipeline's free
sanity check) and the 2021–2025 MCX futures suspension (the data-regime trap
the whole design pivoted around).

    python scripts/make_flagship_chart.py   # -> figures/spot_history.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# palette: validated reference instance (series slot 1 + ink/chrome tokens)
SURFACE = "#fcfcfb"
SERIES = "#2a78d6"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

SUSPENSION = (pd.Timestamp("2021-08-01"), pd.Timestamp("2025-07-29"))
FLOODS = pd.Timestamp("2018-08-15")


def main() -> None:
    # spot_daily_canonical.csv is the mutable file the weekly refresh appends to;
    # spot_daily_full.csv is the quarantine-locked browser-dump original, frozen at
    # 07-Jul-2026. Reading the frozen one silently rendered a month-stale chart.
    spot = pd.read_csv(
        ROOT / "data" / "processed" / "spot_daily_canonical.csv",
        parse_dates=["date"], index_col="date",
    )["spot_avg"]
    weekly = spot.resample("W").mean()
    peak_day, peak = spot.idxmax(), spot.max()

    fig, ax = plt.subplots(figsize=(12.8, 6.4), dpi=150)
    fig.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    ax.axvspan(*SUSPENSION, color=INK, alpha=0.05, zorder=1)
    ax.plot(weekly.index, weekly, color=SERIES, lw=2,
            solid_joinstyle="round", solid_capstyle="round", zorder=3)
    ax.plot(peak_day, peak, "o", ms=8, color=SERIES,
            markeredgecolor=SURFACE, markeredgewidth=2, zorder=4)

    ax.annotate(
        f"Aug-2018 Kerala floods →\npeak auction day ₹{peak:,.0f}/kg",
        xy=(peak_day, peak), xytext=(pd.Timestamp("2015-06-01"), 4050),
        color=INK_2, fontsize=10.5, linespacing=1.4,
        arrowprops=dict(arrowstyle="-", color=BASELINE, lw=1,
                        connectionstyle="arc3,rad=-0.15"),
    )
    mid = SUSPENSION[0] + (SUSPENSION[1] - SUSPENSION[0]) / 2
    ax.text(mid, 4300, "MCX futures suspended\n(relaunched 29 Jul 2025)",
            ha="center", color=INK_2, fontsize=10.5, linespacing=1.4)

    ax.set_title(
        "Indian small cardamom — 12 years of e-auction spot prices",
        color=INK, fontsize=15, fontweight="semibold", loc="left", pad=16,
    )
    # Derived, never hardcoded: the previous literal said 3,148 days through
    # Jul-2026 and silently outlived two dataset refreshes.
    ax.text(0, 1.015, "Weekly mean of Spices Board daily auction averages, "
            f"₹/kg · {len(spot):,} auction days, "
            f"{spot.index.min():%b %Y} – {spot.index.max():%b %Y}",
            transform=ax.transAxes, color=INK_2, fontsize=10.5)

    ax.grid(axis="y", color=GRID, lw=1)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelsize=10, length=0)
    ax.yaxis.set_major_formatter(lambda v, _: f"₹{v:,.0f}")
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_ylim(0, 4700)
    ax.margins(x=0.01)

    out = ROOT / "figures" / "spot_history.png"
    out.parent.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    print(f"saved {out} (peak {peak_day:%d-%b-%Y} at ₹{peak:,.0f}/kg)")


if __name__ == "__main__":
    main()
