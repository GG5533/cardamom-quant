"""Guatemala cardamom exports via UN Comtrade — the cross-market supply signal.

Why: Guatemala is the world's #1 cardamom exporter (~52% of 2024 export
value). When its crop fails — as in 2024-25, down ~40-50% — global buyers
rotate to Indian supply and Indian prices re-rate. No India-only feature set
can see this coming; monthly Guatemalan export volumes can.

Access (verified 2026-07-02): UN Comtrade API v1 is free — without a key,
unlimited calls capped at 500 records each; a free registration key lifts
limits. Endpoint:

    https://comtradeapi.un.org/data/v1/get/C/M/HS
        ?reporterCode=320          (Guatemala)
        &cmdCode=090831,090832     (cardamoms, neither crushed nor ground / other)
        &flowCode=X                (exports)
        &period=YYYYMM[,YYYYMM...]
        &partnerCode=0             (world)

Optional key goes in the 'Ocp-Apim-Subscription-Key' header (env var
COMTRADE_KEY). Response: JSON with a 'data' list; we use netWgt (kg) and
primaryValue (USD).

LEAKAGE DISCIPLINE: national trade statistics publish late. Features carry a
conservative 3-month publication lag — a month-M print is usable from the
first day of month M+3. Late but honest beats early but fictional.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .base import BaseLoader, ValidationError

logger = logging.getLogger(__name__)

COMTRADE_URL = "https://comtradeapi.un.org/data/v1/get/C/M/HS"
GUATEMALA = 320
CARDAMOM_HS = "090831,090832"
PUB_LAG_MONTHS = 3


def parse_comtrade(payload: dict | str) -> pd.DataFrame:
    """Comtrade JSON -> monthly frame [exp_kg, exp_usd], HS codes summed."""
    if isinstance(payload, str):
        payload = json.loads(payload)
    rows = payload.get("data") or []
    if not rows:
        raise ValueError("comtrade: empty 'data' — check period/codes/key")
    df = pd.DataFrame(rows)
    df["month"] = pd.to_datetime(df["period"].astype(str), format="%Y%m")
    g = df.groupby("month")
    out = pd.DataFrame(
        {
            "exp_kg": g["netWgt"].sum(min_count=1),
            "exp_usd": g["primaryValue"].sum(min_count=1),
        }
    ).sort_index()
    out.index.name = "month"
    return out


class GuatemalaExportsLoader(BaseLoader):
    SOURCE = "comtrade_gtm"
    SCHEMA = {
        "exp_kg": "Guatemala cardamom exports, kg/month",
        "exp_usd": "Guatemala cardamom exports, USD/month",
        "exp_kg_yoy": "yoy % change in export volume",
        "supply_shock": "negative yoy z-score vs 3y history (crop-failure gauge)",
    }

    def __init__(
        self,
        raw_dir: Path = config.RAW_DIR / "comtrade",
        processed_dir: Path = config.PROCESSED_DIR,
    ):
        super().__init__(raw_dir, processed_dir)

    # ----------------------------------------------------------------- fetch
    def fetch(self, force: bool = False, start_year: int = 2015) -> int:
        import requests

        headers = {}
        key = os.environ.get("COMTRADE_KEY")
        if key:
            headers["Ocp-Apim-Subscription-Key"] = key
        new = 0
        last_year = pd.Timestamp.today().year
        for year in range(start_year, last_year + 1):
            path = self.raw_dir / f"gtm_{year}.json"
            if path.exists() and not force and year < last_year:
                continue  # closed years never change
            periods = ",".join(f"{year}{m:02d}" for m in range(1, 13))
            r = requests.get(
                COMTRADE_URL,
                params={
                    "reporterCode": GUATEMALA,
                    "cmdCode": CARDAMOM_HS,
                    "flowCode": "X",
                    "period": periods,
                    "partnerCode": 0,
                },
                headers=headers,
                timeout=60,
            )
            r.raise_for_status()
            path.write_text(r.text, encoding="utf-8")
            new += 1
        return new

    # ----------------------------------------------------------------- parse
    def parse(self) -> pd.DataFrame:
        files = sorted(self.raw_dir.glob("gtm_*.json"))
        if not files:
            raise FileNotFoundError(f"no Comtrade payloads in {self.raw_dir}")
        frames = []
        for f in files:
            try:
                frames.append(parse_comtrade(f.read_text(encoding="utf-8")))
            except ValueError as e:
                logger.warning("%s: %s", f.name, e)
        monthly = pd.concat(frames).sort_index()
        monthly = monthly[~monthly.index.duplicated(keep="last")]
        return self.add_features(monthly)

    @staticmethod
    def add_features(monthly: pd.DataFrame, z_window_months: int = 36) -> pd.DataFrame:
        m = monthly.copy()
        m["exp_kg_yoy"] = m["exp_kg"].pct_change(12)
        mu = m["exp_kg_yoy"].rolling(z_window_months, min_periods=24).mean()
        sd = m["exp_kg_yoy"].rolling(z_window_months, min_periods=24).std()
        # supply SHOCK is a shortage gauge: positive when exports collapse
        m["supply_shock"] = -((m["exp_kg_yoy"] - mu) / sd)
        return m

    @staticmethod
    def to_daily_features(
        index: pd.DatetimeIndex,
        monthly: pd.DataFrame,
        pub_lag_months: int = PUB_LAG_MONTHS,
    ) -> pd.DataFrame:
        """Monthly prints mapped to trading days, publication-lagged."""
        lagged = monthly.copy()
        lagged.index = lagged.index + pd.DateOffset(months=pub_lag_months)
        keys = pd.DatetimeIndex(index.to_period("M").to_timestamp())
        out = pd.DataFrame(index=index)
        out.index.name = "date"
        for col in ("exp_kg_yoy", "supply_shock"):
            out[f"gtm_{col}"] = (
                lagged[col].reindex(keys).to_numpy() if col in lagged else np.nan
            )
        return out

    # -------------------------------------------------------------- validate
    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().validate(df)
        if df["exp_kg"].dropna().lt(0).any():
            raise ValidationError("comtrade: negative export volume")
        return df
