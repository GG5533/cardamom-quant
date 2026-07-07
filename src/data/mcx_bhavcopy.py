"""MCXBhavcopyLoader — cardamom futures from MCX Bhavcopy files.

Tier-1 strategy (deliberate): a *drop-folder parser*. The MCX website is a
JS-rendered ASP.NET app that community tools drive with Selenium; endpoints
churn and bot-block. Rather than couple the research pipeline to a fragile
scrape, this loader parses any Bhavcopy file placed in ``data/raw/mcx/`` —
manual bulk download is a bounded one-time task (~230 trading days since the
2025-07-29 relaunch) and the parser carries the engineering weight:

  * handles BOTH schema eras — the classic ``BhavCopyDateWise_YYYYMMDD.csv``
    layout and the post-July-2024 UDiFF common format;
  * filters to CARDAMOM, tags contract regime (pre-2021 vs post-relaunch);
  * builds a continuous front-month series by RETURN SPLICING: each day's
    return is computed on the *same* contract (front contract's close vs its
    own previous-day close), so roll days never inject a phantom P&L. The
    continuous level is the cumulated spliced return anchored at the latest
    close.

Tier-2 (optional, separate): a small AJAX client can be added once the JSON
endpoint used by mcxindia.com/market-data/bhavcopy is confirmed in DevTools.

IMPORTANT: the future was suspended in 2021 and relaunched 2025-07-29 as a
compulsory-delivery contract. The two regimes must never be spliced into one
return series; ``regime`` makes that impossible to do silently.
"""
from __future__ import annotations

import logging
import re
import zipfile
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .base import BaseLoader, ValidationError

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ schemas
# Classic datewise CSV and UDiFF use different headers. Everything is mapped
# onto this canonical set; matching is fuzzy on normalised names so minor
# header drift doesn't break the pipeline.
_CANON = {
    "date": ["date", "traddt", "trade date", "trade_date", "bizdt"],
    "symbol": ["symbol", "tckrsymb", "instrument identifier", "commodity"],
    "expiry": ["expiry date", "expirydt", "xpry dt", "xprydt", "expiry"],
    "open": ["open", "opnpric", "open price"],
    "high": ["high", "hghpric", "high price"],
    "low": ["low", "lwpric", "low price"],
    "close": ["close", "clspric", "close price"],
    "prev_close": ["previous close", "prvsclsgpric", "prev close"],
    "volume": ["volume", "volume(lots)", "ttltradgvol", "volume (lots)"],
    "value": ["value", "ttltrfval", "value (rs. in lakhs)", "value(lakhs)"],
    "oi": ["open interest", "opnintrst", "open interest(lots)", "oi"],
}


def _norm(c: str) -> str:
    return re.sub(r"[^a-z0-9()]+", " ", str(c).lower()).strip()


def _to_date(s: pd.Series) -> pd.Series:
    """Parse dates that may be ISO (UDiFF: 2026-07-01) or Indian
    dd-mm-yyyy (classic: 30-06-2026) within the same backfill.

    ISO strings are matched explicitly so dayfirst never day-swaps them.
    """
    s = s.astype(str).str.strip()
    iso = s.str.match(r"^\d{4}-\d{2}-\d{2}")
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    out[iso] = pd.to_datetime(s[iso], format="%Y-%m-%d", errors="coerce")
    out[~iso] = pd.to_datetime(s[~iso], dayfirst=True, errors="coerce")
    return out


def _map_columns(df: pd.DataFrame) -> pd.DataFrame:
    normed = {_norm(c): c for c in df.columns}
    out = {}
    for canon, aliases in _CANON.items():
        for a in aliases:
            if a in normed:
                out[canon] = df[normed[a]]
                break
    return pd.DataFrame(out)


class MCXBhavcopyLoader(BaseLoader):
    SOURCE = "mcx_cardamom"
    SCHEMA = {
        "fut_close": "front-contract close, Rs/kg",
        "fut_volume": "front-contract volume, lots",
        "fut_oi": "front-contract open interest, lots",
        "contract": "front contract expiry (YYYY-MM-DD)",
        "days_to_expiry": "calendar days to front expiry",
        "fut_ret": "spliced daily return (same-contract, roll-safe)",
        "fut_cont": "continuous level: cumulated fut_ret anchored at last close",
        "regime": "'pre2021' or 'post2025' — never mix across the gap",
    }

    def __init__(
        self,
        raw_dir: Path = config.MCX_RAW_DIR,
        processed_dir: Path = config.PROCESSED_DIR,
        symbol: str = config.MCX_SYMBOL,
    ):
        super().__init__(raw_dir, processed_dir)
        self.symbol = symbol.upper()

    # ----------------------------------------------------------------- fetch
    def fetch(self, force: bool = False) -> int:
        """Tier-1 is manual: files are dropped into raw_dir by the user.

        This just reports what is present so run.py logs are informative.
        """
        n = len(list(self.raw_dir.glob("*.csv"))) + len(list(self.raw_dir.glob("*.zip")))
        if n == 0:
            logger.warning(
                "no Bhavcopy files in %s — download from "
                "https://www.mcxindia.com/market-data/bhavcopy and drop them here",
                self.raw_dir,
            )
        return 0  # fetch never downloads in tier-1

    # ----------------------------------------------------------------- parse
    def _read_any(self, path: Path) -> pd.DataFrame:
        """Read a CSV or a zip containing CSVs."""
        if path.suffix.lower() == ".zip":
            frames = []
            with zipfile.ZipFile(path) as z:
                for name in z.namelist():
                    if name.lower().endswith(".csv"):
                        frames.append(
                            pd.read_csv(StringIO(z.read(name).decode("utf-8", "replace")))
                        )
            return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        return pd.read_csv(path)

    def parse_contracts(self) -> pd.DataFrame:
        """All cardamom contract-day rows across all cached files."""
        files = sorted(list(self.raw_dir.glob("*.csv")) + list(self.raw_dir.glob("*.zip")))
        if not files:
            raise FileNotFoundError(f"no Bhavcopy files in {self.raw_dir}")
        frames = []
        for f in files:
            try:
                raw = self._read_any(f)
            except Exception as e:  # one bad file must not kill the backfill
                logger.error("failed to read %s: %s", f.name, e)
                continue
            if raw.empty:
                continue
            df = _map_columns(raw)
            if "symbol" not in df or "date" not in df:
                logger.warning("%s: unrecognised schema, skipped", f.name)
                continue
            df = df[df["symbol"].astype(str).str.upper().str.strip() == self.symbol]
            if df.empty:
                continue
            df["date"] = _to_date(df["date"])
            df["expiry"] = _to_date(df["expiry"])
            for c in ("open", "high", "low", "close", "prev_close", "volume", "value", "oi"):
                if c in df:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            frames.append(df.dropna(subset=["date", "expiry", "close"]))
        if not frames:
            raise FileNotFoundError(
                f"Bhavcopy files present in {self.raw_dir} but none contained "
                f"{self.symbol} rows"
            )
        out = pd.concat(frames, ignore_index=True)
        out = out.drop_duplicates(subset=["date", "expiry"], keep="last")
        return out.sort_values(["date", "expiry"]).reset_index(drop=True)

    def parse(self) -> pd.DataFrame:
        return self.build_continuous(self.parse_contracts())

    # ------------------------------------------------------------------ roll
    @staticmethod
    def build_continuous(contracts: pd.DataFrame) -> pd.DataFrame:
        """Front selection by max OI (volume, then nearest-expiry tiebreaks),
        continuous series by return splicing.

        Return on day t uses the day-t front contract's own close on t-1
        (looked up in the full contracts table), so a roll never creates a
        spurious jump. Returns across the 2021–2025 suspension gap are NaN
        by construction and by regime guard.
        """
        c = contracts.copy()
        c = c[c["expiry"] >= c["date"]]
        c["_oi"] = c["oi"].fillna(0) if "oi" in c else 0.0
        c["_vol"] = c["volume"].fillna(0) if "volume" in c else 0.0
        c["_dte"] = (c["expiry"] - c["date"]).dt.days
        c = c.sort_values(
            ["date", "_oi", "_vol", "_dte"], ascending=[True, False, False, True]
        )
        front = c.groupby("date").head(1).set_index("date").sort_index()

        # same-contract previous close lookup: (date, expiry) -> close
        px = c.set_index(["date", "expiry"])["close"]
        dates = front.index.to_list()
        rets: list[float] = [np.nan]
        for prev_d, cur_d in zip(dates[:-1], dates[1:]):
            exp = front.loc[cur_d, "expiry"]
            prev_close = px.get((prev_d, exp), np.nan)
            cur_close = front.loc[cur_d, "close"]
            if pd.notna(prev_close) and prev_close > 0:
                rets.append(cur_close / prev_close - 1.0)
            else:  # contract didn't trade yesterday (e.g. first day, gap)
                rets.append(np.nan)

        out = pd.DataFrame(
            {
                "fut_close": front["close"],
                "fut_volume": front["_vol"],
                "fut_oi": front["_oi"],
                "contract": front["expiry"].dt.strftime("%Y-%m-%d"),
                "days_to_expiry": front["_dte"],
                "fut_ret": rets,
            },
            index=front.index,
        )
        out["regime"] = np.where(
            out.index >= pd.Timestamp(config.MCX_RELAUNCH_DATE), "post2025", "pre2021"
        )
        # kill any return that would straddle the regime boundary
        regime_change = pd.Series(out["regime"]).ne(pd.Series(out["regime"]).shift())
        out.loc[regime_change.values, "fut_ret"] = np.nan
        out.iloc[0, out.columns.get_loc("fut_ret")] = np.nan

        # continuous level: anchored at each regime's latest close, walk back
        out["fut_cont"] = np.nan
        for _, block in out.groupby("regime", sort=False):
            level = np.empty(len(block))
            level[-1] = block["fut_close"].iloc[-1]
            r = block["fut_ret"].to_numpy()
            for i in range(len(block) - 2, -1, -1):
                growth = 1.0 + (r[i + 1] if np.isfinite(r[i + 1]) else 0.0)
                level[i] = level[i + 1] / growth if growth != 0 else level[i + 1]
            out.loc[block.index, "fut_cont"] = level
        out.index.name = "date"
        return out

    # -------------------------------------------------------------- validate
    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().validate(df)
        if df["fut_close"].le(0).any():
            raise ValidationError("mcx: non-positive closes")
        gap = df[
            (df.index > config.MCX_SUSPENSION_DATE)
            & (df.index < config.MCX_RELAUNCH_DATE)
        ]
        if not gap.empty:
            raise ValidationError(
                f"mcx: {len(gap)} rows inside the 2021–2025 suspension gap — "
                "check source files"
            )
        # spliced returns should never show a one-day move beyond the widened
        # daily price limit (4% + 2%); allow slack for limit-to-limit moves
        big = df["fut_ret"].abs().gt(0.15).sum()
        if big:
            logger.warning("mcx: %d daily returns exceed 15%% — inspect rolls", big)
        return df
