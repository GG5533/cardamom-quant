"""Unit tests for the real-data ingestion layer.

Everything here runs offline against fixtures — the parsers and the roll /
aggregation / anomaly math are pure functions of files, by design.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FIX = ROOT / "tests" / "fixtures"

from src.data.imd_rainfall import IMDRainfallLoader  # noqa: E402
from src.data.mcx_bhavcopy import MCXBhavcopyLoader  # noqa: E402
from src.data.spices_board import SpicesBoardLoader  # noqa: E402
from src.data.loaders import build_market_dataset  # noqa: E402


# --------------------------------------------------------------- spices board
def test_spices_parse_html():
    html = (FIX / "spices_page.html").read_text()
    rows = SpicesBoardLoader._parse_html(html)
    assert len(rows) == 6
    assert set(rows.columns) >= {"date", "auctioneer", "qty_arrived", "spot_avg"}
    assert rows["spot_avg"].between(2700, 3000).all()
    assert rows["date"].max() == pd.Timestamp("2026-07-01")


def test_spices_daily_aggregation_qty_weighted():
    html = (FIX / "spices_page.html").read_text()
    sessions = SpicesBoardLoader._parse_html(html)
    daily = SpicesBoardLoader.aggregate_daily(sessions)
    assert len(daily) == 3  # 29, 30 Jun, 1 Jul
    d = daily.loc["2026-07-01"]
    # hand-computed qty_sold-weighted average of 2814.97 and 2888.57
    expected = (2814.97 * 62186.9 + 2888.57 * 53612.2) / (62186.9 + 53612.2)
    assert abs(d["spot_avg"] - expected) < 1e-6
    assert d["n_sessions"] == 2
    assert abs(d["qty_arrived"] - (65303.00 + 58137.4)) < 1e-6
    # weighted mean must sit strictly between the two session averages
    assert 2814.97 < d["spot_avg"] < 2888.57


def test_spices_loader_end_to_end(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    shutil.copy(FIX / "spices_page.html", raw / "page_0001.html")
    loader = SpicesBoardLoader(raw_dir=raw, processed_dir=tmp_path / "proc")
    df = loader.validate(loader.parse())
    assert df.index.is_monotonic_increasing
    assert df["spot_avg"].gt(0).all()


def test_spices_validation_rejects_garbage(tmp_path):
    loader = SpicesBoardLoader(raw_dir=tmp_path, processed_dir=tmp_path)
    bad = pd.DataFrame(
        {"spot_avg": [1.0], "spot_max": [2.0], "qty_arrived": [1.0],
         "qty_sold": [1.0], "n_sessions": [1]},
        index=pd.DatetimeIndex(["2026-01-01"], name="date"),
    )
    with pytest.raises(Exception):
        loader.validate(bad)  # price below sanity band


# ------------------------------------------------------------------------ mcx
def _mcx_loader(tmp_path, *fixtures):
    raw = tmp_path / "mcx"
    raw.mkdir(parents=True, exist_ok=True)
    for f in fixtures:
        shutil.copy(FIX / f, raw / f)
    return MCXBhavcopyLoader(raw_dir=raw, processed_dir=tmp_path / "proc")


def test_mcx_parses_both_schema_eras(tmp_path):
    loader = _mcx_loader(
        tmp_path, "BhavCopyDateWise_20260630.csv", "BhavCopy_UDiFF_20260701.csv"
    )
    contracts = loader.parse_contracts()
    # 2 cardamom contracts per day, 2 days; GOLD/SILVER filtered out
    assert len(contracts) == 4
    assert (contracts["symbol"].str.upper() == "CARDAMOM").all()
    assert contracts["date"].nunique() == 2


def test_mcx_front_selection_and_spliced_return(tmp_path):
    loader = _mcx_loader(
        tmp_path, "BhavCopyDateWise_20260630.csv", "BhavCopy_UDiFF_20260701.csv"
    )
    df = loader.parse()
    # front = July contract on both days (higher OI)
    assert (df["contract"] == "2026-07-31").all()
    # spliced return: July close 2971 vs July close 2950 (same contract)
    r = df.loc["2026-07-01", "fut_ret"]
    assert abs(r - (2971.0 / 2950.0 - 1.0)) < 1e-9
    assert np.isnan(df["fut_ret"].iloc[0])  # first day has no lookback
    assert (df["regime"] == "post2025").all()


def test_mcx_continuous_level_no_roll_jump():
    """On a roll day the spliced return must be the NEW contract's own
    day-over-day return, not the front-close jump."""
    contracts = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2025-08-27", "2025-08-27", "2025-08-28", "2025-08-28"]
            ),
            "expiry": pd.to_datetime(
                ["2025-08-29", "2025-09-30", "2025-08-29", "2025-09-30"]
            ),
            "close": [3000.0, 3100.0, 3010.0, 3110.0],
            "volume": [100, 20, 10, 90],
            "oi": [200, 50, 30, 250],  # OI migrates: roll on 08-28
        }
    )
    out = MCXBhavcopyLoader.build_continuous(contracts)
    assert out.loc["2025-08-27", "contract"] == "2025-08-29"
    assert out.loc["2025-08-28", "contract"] == "2025-09-30"
    # return on roll day: Sep contract 3110 vs Sep contract 3100 yesterday
    assert abs(out.loc["2025-08-28", "fut_ret"] - (3110.0 / 3100.0 - 1.0)) < 1e-9
    # naive front-close return would have been 3010/3000-1; ensure we did NOT use it
    assert abs(out.loc["2025-08-28", "fut_ret"] - (3010.0 / 3000.0 - 1.0)) > 1e-4


def test_mcx_regime_guard():
    contracts = pd.DataFrame(
        {
            "date": pd.to_datetime(["2021-06-01", "2025-08-01"]),
            "expiry": pd.to_datetime(["2021-07-15", "2025-08-29"]),
            "close": [1500.0, 2900.0],
            "volume": [10, 50],
            "oi": [20, 100],
        }
    )
    out = MCXBhavcopyLoader.build_continuous(contracts)
    assert out.loc["2021-06-01", "regime"] == "pre2021"
    assert out.loc["2025-08-01", "regime"] == "post2025"
    # no return may straddle the suspension gap
    assert np.isnan(out.loc["2025-08-01", "fut_ret"])


# ------------------------------------------------------------------------ imd
def test_rain_anomaly_math():
    # two years of synthetic rain: wet mid-year, dry winters
    idx = pd.date_range("2020-01-01", "2021-12-31", freq="D")
    doy = idx.dayofyear.to_numpy()
    rain = 10.0 + 8.0 * np.sin((doy - 150) / 366 * 2 * np.pi).clip(0)
    df = pd.DataFrame({"rain_mm": rain}, index=idx)
    out = IMDRainfallLoader.add_anomaly(df)
    assert {"rain_climatology", "rain_anomaly", "source"} <= set(out.columns)
    # with 2 identical years, anomaly ≈ 0 up to smoothing error
    assert out["rain_anomaly"].abs().mean() < 1.0
    # monsoon months wetter than winter in the climatology
    assert (
        out[out.index.month == 7]["rain_climatology"].mean()
        > out[out.index.month == 1]["rain_climatology"].mean()
    )


def test_rain_validation_seasonality_guard(tmp_path):
    loader = IMDRainfallLoader(raw_dir=tmp_path, processed_dir=tmp_path)
    idx = pd.date_range("2022-01-01", "2022-12-31", freq="D")
    flat = pd.DataFrame(
        {
            "rain_mm": 5.0,
            "rain_climatology": 5.0,
            "rain_anomaly": 0.0,
            "source": "imd_gridded",
        },
        index=idx,
    )
    with pytest.raises(Exception):
        loader.validate(flat)  # no monsoon signal -> reject


# -------------------------------------------------------------------- dataset
def test_build_market_dataset_alignment(tmp_path):
    spot_idx = pd.to_datetime(["2026-06-29", "2026-06-30", "2026-07-01"])
    spot = pd.DataFrame(
        {
            "spot_avg": [2890.0, 2850.0, 2849.0],
            "spot_max": [4022.0, 3673.0, 3650.0],
            "qty_arrived": [124927.3, 48219.2, 123440.4],
            "qty_sold": [118231.4, 46301.7, 115799.1],
            "n_sessions": [2, 2, 2],
        },
        index=pd.DatetimeIndex(spot_idx, name="date"),
    )
    fut_idx = pd.to_datetime(["2026-06-30", "2026-07-01", "2026-07-02"])
    futures = pd.DataFrame(
        {
            "fut_close": [2950.0, 2971.0, 2980.0],
            "fut_volume": [120.0, 150.0, 130.0],
            "fut_oi": [340.0, 355.0, 360.0],
            "contract": ["2026-07-31"] * 3,
            "days_to_expiry": [31, 30, 29],
            "fut_ret": [np.nan, 0.00712, 0.00303],
            "fut_cont": [2950.0, 2971.0, 2980.0],
            "regime": ["post2025"] * 3,
        },
        index=pd.DatetimeIndex(fut_idx, name="date"),
    )
    rain = pd.DataFrame(
        {
            "rain_mm": [12.0, 30.0, 8.0, 5.0],
            "rain_climatology": [15.0, 15.0, 15.0, 15.0],
            "rain_anomaly": [-3.0, 15.0, -7.0, -10.0],
            "source": ["imd_gridded"] * 4,
        },
        index=pd.DatetimeIndex(
            pd.to_datetime(["2026-06-29", "2026-06-30", "2026-07-01", "2026-07-02"]),
            name="date",
        ),
    )
    df = build_market_dataset(spot=spot, futures=futures, rain=rain)
    # master calendar = union
    assert pd.Timestamp("2026-06-29") in df.index
    assert pd.Timestamp("2026-07-02") in df.index
    # 07-02 has no auction: spot ffilled with staleness 1, basis still valid
    row = df.loc["2026-07-02"]
    assert row["spot_staleness_days"] == 1
    assert abs(row["basis"] - (2980.0 - 2849.0)) < 1e-9
    # 06-29 has no futures print: basis must be NaN, never zero
    assert np.isnan(df.loc["2026-06-29", "basis"])
    # basis_pct consistent
    assert abs(row["basis_pct"] - row["basis"] / row["spot_avg"]) < 1e-12
