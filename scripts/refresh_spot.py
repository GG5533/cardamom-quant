"""Extend the spot backbone with new auction days — THE lever.

    python scripts/refresh_spot.py            # crawl + verify + rebuild
    python scripts/refresh_spot.py --dry-run  # crawl + verify only

The bootstrap CI on every Sharpe in RESULTS_REAL.md narrows as sqrt(T);
after five feature rounds failed to beat the champion, new auction days
are the one input that reliably buys evidence. This script:

  1. incrementally crawls the Spices Board archive (newest-first pages,
     polite delay, stops at full overlap with the cached sample pages);
  2. VERIFIES every crawled row that overlaps the repaired canonical
     history matches it exactly — the thousands-separator corruption the
     browser dump needed repairing for must never re-enter silently.
     Any mismatch aborts before a single byte is written;
  3. appends only strictly-new sessions to sessions_full_repaired.csv,
     regenerates spot_daily_full.csv, and rebuilds market.parquet the
     canonical way (ONI join + placeholder feed columns + IMD rain merge
     from the cached processed feed).

It never touches rows already in the canonical history.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.climate_indices import parse_oni, to_daily_features  # noqa: E402
from src.data.imd_rainfall import IMDRainfallLoader  # noqa: E402
from src.data.spices_board import SpicesBoardLoader  # noqa: E402

RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
# Lineage: the 07-Jul-2026 browser dump (sessions_full_repaired.csv) is the
# immutable provenance seed — macOS quarantine-locks it against rewriting
# anyway. The refresh pipeline owns sessions_canonical.csv from then on.
SEED = OUT / "sessions_full_repaired.csv"
SESSIONS = OUT / "sessions_canonical.csv"
SPOT_DAILY = OUT / "spot_daily_canonical.csv"
KEY = ["date", "auct_key"]
NUM = ["n_lots", "qty_arrived", "qty_sold", "spot_max", "spot_avg"]


def canon_auctioneer(s: pd.Series) -> pd.Series:
    """Match keys across the site's commas and the repaired dump's semicolons.

    The comma-repair pass rewrote 'Ltd, Kochi' as 'Ltd; Kochi' to keep the
    CSV intact; the live site still prints commas. Same sessions, different
    delimiter — canonicalize punctuation + whitespace before comparing.
    """
    return (s.astype(str)
            .str.replace(r"[;,]", " ", regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip().str.lower())


def rebuild_market(daily: pd.DataFrame) -> pd.DataFrame:
    market = daily.copy()
    market["spot_staleness_days"] = 0
    oni = parse_oni((RAW / "climate" / "oni.ascii.txt").read_text())
    market = market.join(to_daily_features(market.index, oni)[["oni"]])
    for col in ("fut_close", "fut_volume", "fut_oi", "contract", "days_to_expiry",
                "fut_ret", "fut_cont", "regime", "basis", "basis_pct"):
        market[col] = pd.NA if col in ("contract", "regime") else float("nan")
    rain = IMDRainfallLoader().load()
    market = market.join(
        rain[["rain_mm", "rain_climatology", "rain_anomaly"]], how="left"
    )
    return market


def main(dry_run: bool) -> None:
    loader = SpicesBoardLoader()
    n_pages = loader.fetch()
    print(f"crawl: {n_pages} pages fetched")
    crawled = loader.parse_sessions()
    crawled["auct_key"] = canon_auctioneer(crawled["auctioneer"])

    src = SESSIONS if SESSIONS.exists() else SEED
    hist = pd.read_csv(src, parse_dates=["date"])
    hist["auct_key"] = canon_auctioneer(hist["auctioneer"])
    print(f"history: {len(hist)} sessions from {src.name}")

    # ---- verify: overlapping keys must match the repaired history ---------
    merged = crawled.merge(hist, on=KEY, suffixes=("_new", "_hist"), how="inner")
    bad = pd.Series(False, index=merged.index)
    for c in NUM:
        bad |= ~np.isclose(merged[f"{c}_new"], merged[f"{c}_hist"],
                           rtol=1e-6, atol=0.02, equal_nan=True)
    if bad.any():
        print(merged.loc[bad, ["date", "auctioneer"] +
                         [f"{c}_{s}" for c in NUM for s in ("new", "hist")]]
              .head(10).to_string())
        sys.exit(f"ABORT: {bad.sum()} overlapping session(s) disagree with the "
                 "repaired history — comma corruption or a source revision. "
                 "Nothing was written.")
    print(f"verify: {len(merged)} overlapping sessions match the history exactly")

    # ---- append strictly-new sessions --------------------------------------
    keys_hist = set(zip(hist["date"], hist["auct_key"]))
    is_new = ~crawled.apply(lambda r: (r["date"], r["auct_key"]) in keys_hist, axis=1)
    new = crawled[is_new].copy()
    if new.empty:
        print("no new sessions — history already current")
        return
    print(f"new sessions: {len(new)} "
          f"({new['date'].min().date()} -> {new['date'].max().date()})")

    full = (pd.concat([hist, new], ignore_index=True)
            .sort_values(KEY).reset_index(drop=True)
            .drop(columns=["auct_key"]))
    daily = loader.validate(loader.aggregate_daily(full))
    print(f"daily: {len(daily)} auction days through {daily.index.max().date()}")

    if dry_run:
        print("dry-run: nothing written")
        return

    # temp-write + atomic replace: plain overwrites intermittently EPERM on
    # this iCloud-synced tree; rename is always permitted
    def _write(df: pd.DataFrame, path: Path, **kw) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        (df.to_parquet(tmp) if path.suffix == ".parquet"
         else df.to_csv(tmp, **kw))
        os.replace(tmp, path)

    _write(full, SESSIONS, index=False)
    _write(daily, SPOT_DAILY)
    market = rebuild_market(daily)
    _write(market, OUT / "market.parquet")
    print(f"wrote {OUT / 'market.parquet'} ({market.shape[0]} x {market.shape[1]}); "
          "rerun run.py / edge_hunt.py — dataset version bumped")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    main(ap.parse_args().dry_run)
