"""SpicesBoardLoader — small-cardamom e-auction archive (the spot backbone).

The Spices Board publishes every e-auction session (date, auctioneer, lots,
quantity arrived/sold, max & average price) in a paginated, server-rendered
HTML archive:

    https://www.indianspices.com/marketing/price/domestic/daily-price-small.html?page=N

Verified 2026-07-02: 567 pages x 10 rows, newest first, no JavaScript needed.
This is the primary venue for Indian small cardamom, quoted ex-Kerala — the
same pricing basis as the MCX future (ex-Vandanmedu, Idukki).

Two auctioneer sessions typically run per auction day, so the loader exposes
both the session-level table and a daily quantity-weighted aggregate.
"""
from __future__ import annotations

import io
import logging
import re
import time
from pathlib import Path

import pandas as pd

from . import config
from .base import BaseLoader, ValidationError

logger = logging.getLogger(__name__)

# Column names as they appear on the website -> canonical names
_COLMAP = {
    "date of auction": "date",
    "auctioneer": "auctioneer",
    "no.of lots": "n_lots",
    "total qty arrived (kgs)": "qty_arrived",
    "qty sold (kgs)": "qty_sold",
    "maxprice (rs./kg)": "spot_max",
    "avg.price (rs./kg)": "spot_avg",
}


def _normalise(col: str) -> str:
    return re.sub(r"\s+", " ", str(col)).strip().lower()


class SpicesBoardLoader(BaseLoader):
    SOURCE = "spices_board"
    SCHEMA = {
        "spot_avg": "daily qty-weighted average auction price, Rs/kg",
        "spot_max": "max of session max prices, Rs/kg",
        "qty_arrived": "total quantity arrived across sessions, kg",
        "qty_sold": "total quantity sold across sessions, kg",
        "n_sessions": "number of auctioneer sessions that day",
    }

    def __init__(
        self,
        raw_dir: Path = config.RAW_DIR / "spices_board",
        processed_dir: Path = config.PROCESSED_DIR,
        base_url: str = config.SPICES_BOARD_URL,
        delay_s: float = config.SPICES_BOARD_DELAY_S,
    ):
        super().__init__(raw_dir, processed_dir)
        self.base_url = base_url
        self.delay_s = delay_s

    # ----------------------------------------------------------------- fetch
    def fetch(self, force: bool = False, max_pages: int = config.SPICES_BOARD_MAX_PAGES) -> int:
        """Walk the archive pages, saving each as raw HTML.

        Incremental logic: the archive is newest-first, so once a fetched
        page's rows are all already present in the cache we can stop.
        A full backfill simply runs until the site's last page.
        """
        import requests  # local import: parse/tests must work offline

        session = requests.Session()
        session.headers.update(
            {"User-Agent": "cardamom-quant research (contact: samihabbal5@icloud.com)"}
        )
        known = self._cached_keys() if not force else set()
        new_files = 0
        for page in range(1, max_pages + 1):
            path = self.raw_dir / f"page_{page:04d}.html"
            url = self.base_url if page == 1 else f"{self.base_url}?page={page}"
            resp = session.get(url, timeout=config.SPICES_BOARD_TIMEOUT_S)
            resp.raise_for_status()
            rows = self._parse_html(resp.text)
            if rows.empty:
                logger.info("page %d: no table rows — end of archive", page)
                break
            path.write_text(resp.text, encoding="utf-8")
            new_files += 1
            keys = set(zip(rows["date"], rows["auctioneer"]))
            if known and keys <= known:
                logger.info("page %d fully overlaps cache — incremental stop", page)
                break
            time.sleep(self.delay_s)
        return new_files

    def _cached_keys(self) -> set:
        try:
            sess = self.parse_sessions()
            return set(zip(sess["date"], sess["auctioneer"]))
        except (FileNotFoundError, ValueError):
            return set()

    # ----------------------------------------------------------------- parse
    @staticmethod
    def _parse_html(html: str) -> pd.DataFrame:
        """Extract the auction table from one archive page.

        Handles both markups the site has shipped: header cells as <th>
        (read_html yields named columns) and header as a plain first row
        (read_html yields integer columns — seen live since Jul-2026).
        """
        try:
            tables = pd.read_html(io.StringIO(html))
        except ValueError:  # no tables on page
            return pd.DataFrame()
        for t in tables:
            if not ({"date of auction", "auctioneer"}
                    <= {_normalise(c) for c in t.columns}):
                header_row = (
                    len(t) > 1
                    and t.iloc[0].astype(str).map(_normalise)
                        .isin(_COLMAP).sum() >= 4
                )
                if header_row:
                    t = t.copy()
                    t.columns = t.iloc[0]
                    t = t.iloc[1:].reset_index(drop=True)
            cols = {_normalise(c) for c in t.columns}
            if {"date of auction", "auctioneer"} <= cols:
                t = t.rename(columns={c: _COLMAP.get(_normalise(c), _normalise(c)) for c in t.columns})
                t = t[[c for c in _COLMAP.values() if c in t.columns]].copy()
                t["date"] = pd.to_datetime(t["date"], format="%d-%b-%Y", errors="coerce")
                for c in ("n_lots", "qty_arrived", "qty_sold", "spot_max", "spot_avg"):
                    t[c] = pd.to_numeric(t[c], errors="coerce")
                return t.dropna(subset=["date", "spot_avg"])
        return pd.DataFrame()

    def parse_sessions(self) -> pd.DataFrame:
        """Session-level table across all cached pages, deduped."""
        pages = sorted(self.raw_dir.glob("page_*.html"))
        if not pages:
            raise FileNotFoundError(
                f"no cached pages in {self.raw_dir}; run fetch() first"
            )
        frames = [self._parse_html(p.read_text(encoding="utf-8")) for p in pages]
        df = pd.concat([f for f in frames if not f.empty], ignore_index=True)
        df = df.drop_duplicates(subset=["date", "auctioneer"], keep="first")
        return df.sort_values("date").reset_index(drop=True)

    def parse(self) -> pd.DataFrame:
        """Daily aggregate: quantity-weighted average price + supply columns."""
        return self.aggregate_daily(self.parse_sessions())

    @staticmethod
    def aggregate_daily(sessions: pd.DataFrame) -> pd.DataFrame:
        s = sessions.copy()
        # Weight session averages by quantity sold; fall back to arrived.
        w = s["qty_sold"].fillna(s["qty_arrived"]).clip(lower=0.0)
        s["_w"] = w.where(w > 0, 1.0)  # degenerate session -> equal weight
        s["_wp"] = s["spot_avg"] * s["_w"]
        g = s.groupby("date")
        daily = pd.DataFrame(
            {
                "spot_avg": g["_wp"].sum() / g["_w"].sum(),
                "spot_max": g["spot_max"].max(),
                "qty_arrived": g["qty_arrived"].sum(min_count=1),
                "qty_sold": g["qty_sold"].sum(min_count=1),
                "n_sessions": g.size(),
            }
        ).sort_index()
        daily.index.name = "date"
        return daily

    # -------------------------------------------------------------- validate
    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().validate(df)
        if df["spot_avg"].le(0).any():
            raise ValidationError("spices_board: non-positive prices")
        # Sanity band: small cardamom has traded roughly Rs 300–5,000/kg over
        # the past decade. Values outside 50–20,000 indicate a parse bug,
        # not a market move.
        if df["spot_avg"].lt(50).any() or df["spot_avg"].gt(20_000).any():
            raise ValidationError("spices_board: price outside sanity band 50–20,000 Rs/kg")
        if (df["qty_sold"] > df["qty_arrived"] * 1.05).any():  # 5% tolerance
            logger.warning("spices_board: qty_sold > qty_arrived on some days")
        return df
