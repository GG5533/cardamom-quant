"""One-command real-data build.

Usage:
    python scripts/build_dataset.py            # parse cached raw files only
    python scripts/build_dataset.py --refresh  # also fetch new data

Spices Board and IMD fetch over the network; MCX is tier-1 manual — drop
Bhavcopy CSVs/zips into data/raw/mcx/ (see src/data/mcx_bhavcopy.py).
The script degrades gracefully: whatever feeds exist are built and the
aligned dataset is written to data/processed/market.parquet (+ .csv).
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import config  # noqa: E402
from src.data.loaders import (  # noqa: E402
    IMDRainfallLoader,
    MCXBhavcopyLoader,
    SpicesBoardLoader,
    build_market_dataset,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("build_dataset")


def main(refresh: bool) -> None:
    spot = futures = rain = None

    sb = SpicesBoardLoader()
    if refresh:
        log.info("Spices Board: fetching (one-time backfill takes ~20 min politely)")
        sb.fetch()
    try:
        spot = sb.load()
        log.info("spot: %d days, %s -> %s", len(spot), spot.index.min().date(),
                 spot.index.max().date())
    except FileNotFoundError as e:
        log.error("spot unavailable: %s", e)
        sys.exit(1)  # spot is the backbone — nothing works without it

    try:
        rain_loader = IMDRainfallLoader()
        if refresh:
            rain_loader.fetch()
        rain = rain_loader.load()
        log.info("rain: %d days", len(rain))
    except Exception as e:  # imdlib missing / not yet downloaded
        log.warning("rain unavailable (%s) — continuing without weather", e)
        import pandas as pd
        rain = pd.DataFrame(
            columns=["rain_mm", "rain_climatology", "rain_anomaly", "source"],
            index=pd.DatetimeIndex([], name="date"),
        )

    try:
        futures = MCXBhavcopyLoader().load()
        log.info("futures: %d days, regimes: %s", len(futures),
                 futures["regime"].value_counts().to_dict())
    except FileNotFoundError:
        log.warning("no MCX Bhavcopy files in %s — spot-only dataset",
                    config.MCX_RAW_DIR)

    df = build_market_dataset(spot=spot, futures=futures, rain=rain)
    out = config.PROCESSED_DIR / "market"
    try:
        df.to_parquet(out.with_suffix(".parquet"))
    except ImportError:
        pass
    df.to_csv(out.with_suffix(".csv"))
    log.info("wrote %s rows x %s cols -> %s.csv", *df.shape, out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="fetch new data first")
    main(ap.parse_args().refresh)
