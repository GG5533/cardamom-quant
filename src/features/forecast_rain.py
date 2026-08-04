"""AS-ISSUED forecast rain — T13's single feature block.

Built from NOAA GEFS v12 reforecast data (`scripts/gefs_backfill_reforecast.py`,
piloted in `scripts/gefs_pilot.py`): for auction day D, `forecast_5d_precip_mm`
is the model's own forecast, issued at D's 00z init, of accumulated
precipitation over the Idukki box across leads 0-120h (i.e. the 5 days
starting at D). This is fundamentally different from the realized-rain
features already in CORE_FEATURE_COLS (`rain_anom_30/90`): those are
trailing rolling means of rainfall that has ALREADY happened, so they only
describe backward-looking climate state. This feature is a forecast of the
same window the 5-day label measures — forward-looking at the moment of
the trading decision, by construction.

Coverage caveat (read before drawing conclusions from anything downstream):
the CSV covers ONLY the NOAA GEFS reforecast era, 2014-11-07 -> 2019-12-31
(1341/1341 auction days in that era). The 2020-present leg uses a different
code path (operational GEFS archive with .idx byte-range subsetting) and
has not been built, so this feature has ZERO coverage from 2020 onward —
entirely disjoint from the live-trading window. Nothing evaluated with
this feature says anything about current predictive power; see T13 in
RESULTS_REAL.md and `scripts/forecast_rain_trial.py` for how that
limitation is handled in the trial itself.

Leakage discipline: the raw CSV value at date D is, by construction,
information available AT D — it is NOAA's forecast product, downloaded
as-is, never touching realized outcomes after D. Unlike realized-rain
aggregates (which need a rolling window + shift(1) to avoid using a day's
own not-yet-closed rainfall), there is nothing to "wait out" here. The
shift(1) applied below is therefore a *pipeline-consistency* choice, not a
leakage fix: every other feature in `build_features()` is contracted to be
"available at the close of t-1", and this feature is made to match that
contract rather than exploit its (arguably legitimate) same-day edge. See
tests/test_signal_layer.py::test_forecast_rain_is_strictly_causal and
::test_forecast_rain_uses_prior_day_value for the causality proof.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data import config

FORECAST_RAIN_FEATURE_COLS = ["fcst_rain_5d"]


def load_forecast_rain(path: Path = config.GEFS_FORECAST_RAIN_CSV) -> pd.Series:
    """Raw as-issued 5-day accumulated precip forecast, indexed by init date.

    Returns an empty-but-typed Series if the CSV hasn't been built yet
    (mirrors the other loaders' "feed not wired -> NaN" convention rather
    than crashing the pipeline).
    """
    path = Path(path)
    if not path.exists():
        return pd.Series(dtype=float, name="forecast_5d_precip_mm")
    df = pd.read_csv(path, dtype={"date": str})
    idx = pd.to_datetime(df["date"], format="%Y%m%d")
    s = pd.Series(
        df["forecast_5d_precip_mm"].to_numpy(dtype=float),
        index=idx,
        name="forecast_5d_precip_mm",
    )
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s


def build_forecast_rain_features(
    market: pd.DataFrame, path: Path = config.GEFS_FORECAST_RAIN_CSV
) -> pd.DataFrame:
    """One feature: `fcst_rain_5d`, aligned to market's calendar and shifted
    1 day to match every other feature's "available at close of t-1"
    contract (see module docstring for why the shift is conservatism, not
    a leakage fix). Rows outside the reforecast era are NaN by
    construction — that NaN IS the coverage caveat, not a bug.
    """
    raw = load_forecast_rain(path)
    aligned = raw.reindex(market.index)
    df = pd.DataFrame(index=market.index)
    df["fcst_rain_5d"] = aligned.shift(1)
    return df
