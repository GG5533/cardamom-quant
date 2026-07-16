"""Tamper-evident forecast ledger — the prospective-validation backbone.

A backtest can always be accused of hindsight; a forecast logged before its
outcome exists cannot. Two append-only CSVs under data/live/:

  forecasts.csv  one row per (auction_date) forecast, written BEFORE the
                 5d outcome exists, hash-chained: each row carries
                 sha256(prev_hash + canonical_payload), so any later edit
                 or deletion breaks every subsequent hash.
  outcomes.csv   one row per matured forecast (>= HORIZON auction days
                 later), scored against the canonical daily file. Also
                 hash-chained, separately.

Nothing here ever rewrites a row. `verify_chain` re-derives every hash and
is run by the test suite and by scripts/forecast.py on every invocation —
the ledger polices itself.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from ..data import config

LIVE_DIR = config.PROCESSED_DIR.parent / "live"
FORECASTS = LIVE_DIR / "forecasts.csv"
OUTCOMES = LIVE_DIR / "outcomes.csv"
GENESIS = "cardamom-quant-forecast-ledger-genesis"


def _canon(payload: dict, fields: list[str]) -> str:
    return "|".join(str(payload[f]) for f in fields)


def _chain_hash(prev_hash: str, canon: str) -> str:
    return hashlib.sha256(f"{prev_hash}|{canon}".encode()).hexdigest()[:16]


class ChainedCsv:
    """Append-only CSV where every row extends a hash chain."""

    def __init__(self, path: Path, fields: list[str]):
        self.path = path
        self.fields = fields

    def read(self) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame(columns=self.fields + ["hash"])
        return pd.read_csv(self.path, dtype=str)

    def last_hash(self) -> str:
        df = self.read()
        return GENESIS if df.empty else str(df["hash"].iloc[-1])

    def append(self, payload: dict) -> str:
        missing = set(self.fields) - set(payload)
        if missing:
            raise ValueError(f"ledger payload missing {sorted(missing)}")
        h = _chain_hash(self.last_hash(), _canon(payload, self.fields))
        row = {**{f: payload[f] for f in self.fields}, "hash": h}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        header = not self.path.exists()
        pd.DataFrame([row]).to_csv(self.path, mode="a", header=header, index=False)
        return h

    def verify_chain(self) -> int:
        """Recompute every hash; raise on the first broken link."""
        df = self.read()
        prev = GENESIS
        for i, row in df.iterrows():
            expect = _chain_hash(prev, _canon(row, self.fields))
            if row["hash"] != expect:
                raise ValueError(
                    f"{self.path.name}: hash chain broken at row {i} "
                    f"({row.get('auction_date', '?')}) — ledger was edited"
                )
            prev = row["hash"]
        return len(df)


FORECAST_FIELDS = [
    "auction_date",     # feature-row date the forecast is made from
    "made_on_utc",      # wall-clock time the row was written
    "p_up",             # calibrated P(5d forward return > 0)
    "signal",           # 2p-1, the position direction/size input
    "spot_avg",         # price level at forecast time (context)
    "model",            # config tag, e.g. champion-physics-gbm
    "git_sha",          # code version that produced it
]
OUTCOME_FIELDS = [
    "auction_date",     # matches a forecasts.csv row
    "scored_on_utc",
    "fwd_5d_ret",       # realized forward return
    "outcome_up",       # 1.0 / 0.0
    "hit",              # forecast direction matched
    "brier",            # (p_up - outcome)^2
]


def forecast_ledger() -> ChainedCsv:
    return ChainedCsv(FORECASTS, FORECAST_FIELDS)


def outcome_ledger() -> ChainedCsv:
    return ChainedCsv(OUTCOMES, OUTCOME_FIELDS)
