"""Central configuration for the real-data ingestion layer.

Everything environment-specific lives here so loaders stay pure.
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"            # immutable download cache — never edited
PROCESSED_DIR = DATA_DIR / "processed"  # parquet outputs of each loader

# ---------------------------------------------------------------------------
# Spices Board e-auction archive (spot backbone)
# ---------------------------------------------------------------------------
SPICES_BOARD_URL = (
    "https://www.indianspices.com/marketing/price/domestic/daily-price-small.html"
)
SPICES_BOARD_DELAY_S = 1.5      # polite delay between page requests
SPICES_BOARD_TIMEOUT_S = 30
SPICES_BOARD_MAX_PAGES = 1000   # hard stop; archive was 567 pages on 2026-07-02

# ---------------------------------------------------------------------------
# MCX Bhavcopy
# ---------------------------------------------------------------------------
MCX_RAW_DIR = RAW_DIR / "mcx"   # drop BhavCopyDateWise_*.csv / UDiFF files here
MCX_SYMBOL = "CARDAMOM"
# Contract regimes: the future was suspended in 2021 and relaunched 2025-07-29
# as a compulsory-delivery contract. Features must never cross the gap.
MCX_RELAUNCH_DATE = "2025-07-29"
MCX_SUSPENSION_DATE = "2021-12-31"  # conservative end of old-regime data

# ---------------------------------------------------------------------------
# IMD rainfall (Idukki cardamom belt)
# ---------------------------------------------------------------------------
IMD_RAW_DIR = RAW_DIR / "imd"
# Bounding box around the Kerala cardamom hill tract (Idukki district)
IDUKKI_LAT = (9.4, 10.2)
IDUKKI_LON = (76.7, 77.4)
IMD_FIRST_YEAR = 2010           # backfill start; gridded data exists from 1901
