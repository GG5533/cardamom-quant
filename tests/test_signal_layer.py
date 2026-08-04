"""Tests for the cross-market/macro/microstructure signal layer (offline)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.comtrade import GuatemalaExportsLoader, parse_comtrade  # noqa: E402
from src.data.fx import parse_fred_csv, to_daily_features as fx_daily  # noqa: E402
from src.features.alt_features import (  # noqa: E402
    ALT_FEATURE_COLS,
    auction_microstructure,
    build_alt_features,
)
from src.features.forecast_rain import (  # noqa: E402
    build_forecast_rain_features,
    load_forecast_rain,
)

COMTRADE_FIXTURE = json.dumps(
    {
        "data": [
            {"period": 202410, "cmdCode": "090831", "netWgt": 3_000_000, "primaryValue": 90_000_000},
            {"period": 202410, "cmdCode": "090832", "netWgt": 200_000, "primaryValue": 5_000_000},
            {"period": 202411, "cmdCode": "090831", "netWgt": 2_500_000, "primaryValue": 80_000_000},
        ]
    }
)

FRED_FIXTURE = """DATE,DEXINUS
2026-06-25,86.10
2026-06-26,86.25
2026-06-29,.
2026-06-30,86.40
"""


# ------------------------------------------------------------------ comtrade
def test_parse_comtrade_sums_hs_codes():
    m = parse_comtrade(COMTRADE_FIXTURE)
    assert m.loc["2024-10-01", "exp_kg"] == 3_200_000
    assert m.loc["2024-11-01", "exp_usd"] == 80_000_000
    assert len(m) == 2


def test_supply_shock_positive_on_collapse():
    # 4 stable years then a 45% crop failure (the 2024-25 scenario)
    months = pd.date_range("2020-01-01", "2024-12-01", freq="MS")
    kg = pd.Series(1_000_000.0, index=months)
    kg.iloc[-6:] = 550_000.0
    monthly = pd.DataFrame({"exp_kg": kg, "exp_usd": kg * 30})
    feats = GuatemalaExportsLoader.add_features(monthly)
    assert feats["supply_shock"].iloc[-1] > 1.0  # strong shortage signal
    assert abs(feats["exp_kg_yoy"].iloc[-1] - (-0.45)) < 1e-9


def test_comtrade_publication_lag():
    months = pd.date_range("2024-01-01", "2024-12-01", freq="MS")
    monthly = pd.DataFrame(
        {
            "exp_kg": 1.0,
            "exp_usd": 1.0,
            "exp_kg_yoy": np.linspace(0, 1, len(months)),
            "supply_shock": 0.5,
        },
        index=months,
    )
    idx = pd.date_range("2024-01-01", "2025-03-31", freq="D")
    daily = GuatemalaExportsLoader.to_daily_features(idx, monthly)
    # Jan-2024 print (yoy=0.0) usable from Apr-2024, not before
    assert np.isnan(daily.loc["2024-03-15", "gtm_exp_kg_yoy"])
    assert daily.loc["2024-04-15", "gtm_exp_kg_yoy"] == 0.0


# ----------------------------------------------------------------------- fx
def test_parse_fred_csv_missing_dot():
    s = parse_fred_csv(FRED_FIXTURE)
    assert len(s) == 3  # '.' dropped
    assert s[pd.Timestamp("2026-06-30")] == 86.40


def test_fx_features_ffill_limited():
    s = parse_fred_csv(FRED_FIXTURE)
    idx = pd.date_range("2026-06-25", "2026-06-30", freq="D")
    f = fx_daily(idx, s)
    assert f.loc["2026-06-28", "usdinr"] == 86.25  # weekend ffill
    assert f.loc["2026-06-30", "usdinr"] == 86.40


# ------------------------------------------------------------ microstructure
def _spot(n=400, seed=7):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-01", periods=n)
    arrived = pd.Series(rng.uniform(40_000, 120_000, n), index=idx)
    sold = arrived * rng.uniform(0.85, 1.0, n)
    return pd.DataFrame(
        {
            "spot_avg": rng.uniform(2000, 3000, n),
            "spot_max": 3500.0,
            "qty_arrived": arrived,
            "qty_sold": sold,
            "n_sessions": 2,
        },
        index=idx,
    )


def test_microstructure_is_strictly_causal():
    """Mutating TODAY's auction must not change TODAY's features."""
    spot = _spot()
    base = auction_microstructure(spot)
    mutated = spot.copy()
    mutated.iloc[-1, mutated.columns.get_loc("qty_sold")] = 1.0  # crash today
    after = auction_microstructure(mutated)
    last = spot.index[-1]
    for col in ("tension", "tension_z_63", "arrivals_surprise"):
        b, a = base.loc[last, col], after.loc[last, col]
        assert (np.isnan(b) and np.isnan(a)) or b == a


def test_tension_bounded_and_sane():
    f = auction_microstructure(_spot())
    t = f["tension"].dropna()
    assert t.between(0, 1.5).all()
    assert 0.8 < t.mean() < 1.0  # ~90% sell-through in the fixture


# --------------------------------------------------------------------- build
def test_build_alt_features_schema_stable_without_optional_feeds():
    f = build_alt_features(_spot())
    assert list(f.columns) == ALT_FEATURE_COLS
    assert f["oni"].isna().all()          # not wired -> explicit NaN
    assert f["gtm_supply_shock"].isna().all()
    assert f["ramadan_stocking"].notna().all()  # calendars always present


def test_build_alt_features_with_climate():
    oni = pd.Series(
        {pd.Timestamp(2024, m, 1): 0.1 * m for m in range(1, 13)}, name="oni"
    )
    f = build_alt_features(_spot(), oni=oni)
    assert f["oni"].notna().sum() > 0
    assert list(f.columns) == ALT_FEATURE_COLS


# ------------------------------------------------------------ forecast rain
def _market_for_forecast(n=40, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    return pd.DataFrame({"spot_avg": rng.uniform(2000, 3000, n)}, index=idx)


def _forecast_csv(tmp_path, dates, values):
    path = tmp_path / "gefs_forecast_rain.csv"
    pd.DataFrame(
        {
            "date": dates,
            "forecast_5d_precip_mm": values,
            "source": "reforecast",
            "fetched_at_utc": "2026-01-01T00:00:00+00:00",
        }
    ).to_csv(path, index=False)
    return path


def test_forecast_rain_is_strictly_causal(tmp_path):
    """Mutating TODAY's as-issued forecast must not change TODAY's feature.

    The raw CSV value at date D is, by construction, a forecast issued AT D
    (never computed from realized outcomes after D) — but build_features()'s
    pipeline-wide contract is "available at the close of t-1", so the
    feature is shift(1)'d. This proves that shift actually holds: day t's
    OWN same-day forecast must not leak into day t's feature.
    """
    market = _market_for_forecast()
    dates = [d.strftime("%Y%m%d") for d in market.index]
    values = np.linspace(0.0, 200.0, len(dates))
    path = _forecast_csv(tmp_path, dates, values)

    base = build_forecast_rain_features(market, path=path)

    t = market.index[10]
    mutated = values.copy()
    mutated[10] = 9999.0  # blow up TODAY's own forecast
    path2 = _forecast_csv(tmp_path, dates, mutated)
    after = build_forecast_rain_features(market, path=path2)

    b, a = base.loc[t, "fcst_rain_5d"], after.loc[t, "fcst_rain_5d"]
    assert (np.isnan(b) and np.isnan(a)) or b == a


def test_forecast_rain_uses_prior_day_value(tmp_path):
    """The shift is a real shift, not an accidental no-op: mutating day t's
    forecast MUST move day t+1's feature, and move it to exactly that
    value — proving fcst_rain_5d(t+1) == raw_forecast(t)."""
    market = _market_for_forecast()
    dates = [d.strftime("%Y%m%d") for d in market.index]
    values = np.linspace(0.0, 200.0, len(dates))
    mutated = values.copy()
    mutated[10] = 4321.0
    path = _forecast_csv(tmp_path, dates, mutated)

    feats = build_forecast_rain_features(market, path=path)
    t_next = market.index[11]
    assert feats.loc[t_next, "fcst_rain_5d"] == 4321.0


def test_forecast_rain_out_of_era_is_nan(tmp_path):
    """Coverage caveat enforced in the data itself: dates the reforecast-era
    CSV never covered come back NaN, never silently zero-filled."""
    market = _market_for_forecast(n=60)
    covered = market.index[:20]  # only the first 20 days have a forecast
    dates = [d.strftime("%Y%m%d") for d in covered]
    values = np.full(len(dates), 25.0)
    path = _forecast_csv(tmp_path, dates, values)

    feats = build_forecast_rain_features(market, path=path)
    assert feats.loc[market.index[30]:, "fcst_rain_5d"].isna().all()
    # shift(1) loses day 0 (no prior value) but gains day 20 (inherits day
    # 19's covered value) -> coverage count is preserved, just shifted.
    assert feats["fcst_rain_5d"].notna().sum() == 20
    assert np.isnan(feats.loc[market.index[0], "fcst_rain_5d"])


def test_load_forecast_rain_missing_file_returns_empty_series(tmp_path):
    """Feed-not-wired convention: missing CSV -> empty typed Series, not a
    crash (mirrors every other loader's 'not yet wired -> NaN' behaviour)."""
    s = load_forecast_rain(path=tmp_path / "does_not_exist.csv")
    assert s.empty
    assert s.dtype == float
