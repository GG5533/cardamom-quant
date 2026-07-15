"""Merge IMD rain features into the canonical market.parquet, in place.

    python scripts/merge_rain.py

Touches ONLY the three rain columns the ingest script reserved as NaN
(rain_mm, rain_climatology, rain_anomaly); spot, ONI and futures columns
are untouched. The pre-merge parquet is versioned in git — `git checkout
-- data/processed/market.parquet` reverts the wiring.

The gridded product ends at the last complete year, so recent months stay
NaN — that is the feed's real publication lag, not a bug to fill.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.loaders import IMDRainfallLoader  # noqa: E402

RAIN_COLS = ["rain_mm", "rain_climatology", "rain_anomaly"]


def main() -> None:
    path = ROOT / "data" / "processed" / "market.parquet"
    market = pd.read_parquet(path)
    already = market["rain_anomaly"].notna().sum()

    rain = IMDRainfallLoader().load()
    aligned = rain[RAIN_COLS].reindex(market.index)
    market[RAIN_COLS] = aligned

    n = market["rain_anomaly"].notna().sum()
    lo = market.index[market["rain_anomaly"].notna()]
    print(f"rain coverage: {already} -> {n} of {len(market)} auction days "
          f"({lo.min().date()} -> {lo.max().date()})")
    market.to_parquet(path)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
