"""BaseLoader: the common contract every real-data feed implements.

Design principles (mirrors the rest of cardamom-quant):
  * raw payloads are cached immutably under data/raw/<source>/ and never
    re-downloaded or edited — reproducibility first;
  * parse() is a pure function of cached raw files, so a broken website
    can never corrupt an existing dataset;
  * validate() fails loudly. A loader that returns silently-wrong data is
    worse than one that crashes.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class ValidationError(RuntimeError):
    """Raised when a loader's output violates its schema contract."""


class BaseLoader(ABC):
    """Fetch → parse → validate → load, with an immutable raw cache."""

    #: canonical output columns, defined by each subclass
    SCHEMA: dict[str, str] = {}
    #: source name, used for cache directories and logging
    SOURCE: str = "base"

    def __init__(self, raw_dir: Path, processed_dir: Path):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ api
    @abstractmethod
    def fetch(self, force: bool = False) -> int:
        """Download raw payloads into ``self.raw_dir``.

        Returns the number of new raw files written. Must be incremental:
        existing files are never re-downloaded unless ``force``.
        """

    @abstractmethod
    def parse(self) -> pd.DataFrame:
        """Parse all cached raw files into the canonical tidy frame."""

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generic schema checks; subclasses add source-specific rules."""
        missing = set(self.SCHEMA) - set(df.columns)
        if missing:
            raise ValidationError(f"{self.SOURCE}: missing columns {sorted(missing)}")
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValidationError(f"{self.SOURCE}: index must be a DatetimeIndex")
        if not df.index.is_monotonic_increasing:
            raise ValidationError(f"{self.SOURCE}: index not sorted")
        return df

    def load(self, refresh: bool = False) -> pd.DataFrame:
        """Cached-or-fetched, parsed, validated frame; also writes parquet."""
        if refresh:
            n = self.fetch()
            logger.info("%s: fetched %d new raw files", self.SOURCE, n)
        df = self.validate(self.parse())
        out = self.processed_dir / f"{self.SOURCE}.parquet"
        try:
            df.to_parquet(out)
        except ImportError:  # pyarrow not installed — parquet cache is optional
            logger.warning("%s: pyarrow missing, skipping parquet cache", self.SOURCE)
        return df
