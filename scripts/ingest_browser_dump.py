"""Ingest the browser-crawled sessions CSV -> canonical daily + market.parquet."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.spices_board import SpicesBoardLoader
from src.data.climate_indices import parse_oni, to_daily_features

RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"


def main(sessions_csv: str) -> None:
    s = pd.read_csv(sessions_csv)
    s.columns = ["date", "auctioneer", "n_lots", "qty_arrived", "qty_sold", "spot_max", "spot_avg"]
    s["date"] = pd.to_datetime(s["date"], format="%d-%b-%Y", errors="coerce")
    for c in ("n_lots", "qty_arrived", "qty_sold", "spot_max", "spot_avg"):
        s[c] = pd.to_numeric(s[c], errors="coerce")
    s = s.dropna(subset=["date", "spot_avg"]).drop_duplicates(["date", "auctioneer"])
    glitch = ~s["spot_avg"].between(50, 20_000)
    if glitch.any():
        print(f"dropping {glitch.sum()} glitch sessions (price outside 50-20000)")
        s = s[~glitch]
    s = s.sort_values("date").reset_index(drop=True)
    print(f"sessions: {len(s)}  {s.date.min().date()} -> {s.date.max().date()}")

    loader = SpicesBoardLoader(raw_dir=RAW / "spices_board", processed_dir=OUT)
    daily = loader.validate(loader.aggregate_daily(s))
    print(f"daily: {len(daily)} auction days")

    market = daily.copy()
    market["spot_staleness_days"] = 0
    oni = parse_oni((RAW / "climate" / "oni.ascii.txt").read_text())
    market = market.join(to_daily_features(market.index, oni)[["oni"]])
    for col in ("fut_close","fut_volume","fut_oi","contract","days_to_expiry",
                "fut_ret","fut_cont","regime","basis","basis_pct",
                "rain_mm","rain_climatology","rain_anomaly"):
        market[col] = pd.NA if col in ("contract","regime") else float("nan")
    market.to_parquet(OUT / "market.parquet")
    s.to_csv(OUT / "spices_sessions_real.csv", index=False)
    daily.to_csv(OUT / "spot_daily_real.csv")
    print(f"wrote {OUT/'market.parquet'}  ({market.shape[0]} x {market.shape[1]})")


if __name__ == "__main__":
    main(sys.argv[1])
