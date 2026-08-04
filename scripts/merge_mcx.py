"""Merge MCX Bhavcopy futures + basis into the canonical market.parquet,
in place.

    python scripts/merge_mcx.py

Touches ONLY the MCX/basis columns the ingest schema reserved as NaN
(fut_close, fut_volume, fut_oi, contract, days_to_expiry, fut_ret, fut_cont,
regime, basis, basis_pct); spot, ONI and rain columns are untouched. The
pre-merge parquet is versioned in git — `git checkout --
data/processed/market.parquet` reverts the wiring.

Basis is computed with the SAME honesty rule as
`src/data/loaders.py::build_market_dataset` (kept in sync deliberately,
not re-imported, because that function also owns the initial join and we
do not want a refactor touching earlier, already-published columns): a
day only gets a basis value where a futures close AND a spot print within
`max_spot_staleness` days both exist. The 2025-07-29 relaunch means basis
is NaN for ~90% of the 11-year history by construction — that is the
feed's real coverage, not a bug to fill.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.mcx_bhavcopy import MCXBhavcopyLoader  # noqa: E402

MAX_SPOT_STALENESS = 3
FUT_COLS = [
    "fut_close", "fut_volume", "fut_oi", "contract", "days_to_expiry",
    "fut_ret", "fut_cont", "regime",
]


def main() -> None:
    path = ROOT / "data" / "processed" / "market.parquet"
    market = pd.read_parquet(path)
    already = market["basis_pct"].notna().sum()

    futures = MCXBhavcopyLoader().load()
    aligned = futures[FUT_COLS].reindex(market.index)
    market[FUT_COLS] = aligned

    ok = (
        market["fut_close"].notna()
        & market["spot_avg"].notna()
        & (market["spot_staleness_days"] <= MAX_SPOT_STALENESS)
    )
    market["basis"] = np.where(ok, market["fut_close"] - market["spot_avg"], np.nan)
    market["basis_pct"] = np.where(ok, market["basis"] / market["spot_avg"], np.nan)

    n = market["basis_pct"].notna().sum()
    cov = market.index[market["basis_pct"].notna()]
    print(f"MCX basis coverage: {already} -> {n} of {len(market)} auction days "
          f"({cov.min().date()} -> {cov.max().date()})")
    market.to_parquet(path)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
