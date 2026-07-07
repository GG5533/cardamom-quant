"""Demand-calendar features — the moving seasonals that day-of-year misses.

The out-of-the-box insight: cardamom demand is driven by two calendars that
DRIFT against the Gregorian year, so the sin/cos day-of-year seasonality in
features/engineering.py structurally cannot capture them:

  * RAMADAN (Gulf demand). Saudi Arabia is the largest importer of Indian
    cardamom (gahwa). Gulf retailers stock 6–8 weeks before Ramadan and
    grocery spend rises ~30% in the two weeks prior — and Ramadan moves
    ~11 days EARLIER every Gregorian year. A price-relevant seasonal that
    rotates through the calendar is invisible to day-of-year features and
    is exactly the kind of signal that differentiates this model.

  * DIWALI (domestic demand). Indian festival buying (sweets, gifting)
    peaks in the weeks before Diwali, which oscillates mid-Oct to mid-Nov
    with the lunisolar calendar.

Leakage note: these are the only "future-dated" inputs the model is allowed —
calendars are known ex ante with certainty, so a feature like "days until
Ramadan" is legitimately available at prediction time. That asymmetry
(calendars yes, prices no) is worth a line in the README.

Dates are tabulated (Umm al-Qura / Indian civil festival calendars). Actual
observance can shift ±1 day on moon sighting; all features here use
multi-week windows, so a one-day tabulation error is immaterial by design.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Ramadan first day (Gregorian), tabulated
RAMADAN_START = {
    2010: "2010-08-11", 2011: "2011-08-01", 2012: "2012-07-20",
    2013: "2013-07-09", 2014: "2014-06-28", 2015: "2015-06-18",
    2016: "2016-06-06", 2017: "2017-05-27", 2018: "2018-05-16",
    2019: "2019-05-06", 2020: "2020-04-24", 2021: "2021-04-13",
    2022: "2022-04-02", 2023: "2023-03-23", 2024: "2024-03-11",
    2025: "2025-03-01", 2026: "2026-02-18", 2027: "2027-02-08",
    2028: "2028-01-28",
}

# Diwali (Lakshmi Puja) day, tabulated
DIWALI = {
    2010: "2010-11-05", 2011: "2011-10-26", 2012: "2012-11-13",
    2013: "2013-11-03", 2014: "2014-10-23", 2015: "2015-11-11",
    2016: "2016-10-30", 2017: "2017-10-19", 2018: "2018-11-07",
    2019: "2019-10-27", 2020: "2020-11-14", 2021: "2021-11-04",
    2022: "2022-10-24", 2023: "2023-11-12", 2024: "2024-11-01",
    2025: "2025-10-20", 2026: "2026-11-08",
}


def _days_to_next(index: pd.DatetimeIndex, events: dict[int, str]) -> pd.Series:
    """Signed day count to the next occurrence of an annual event.

    Positive = event is ahead; 0 = event day. Uses the tabulated dates, so
    values are exact within tabulation accuracy.
    """
    dates = pd.DatetimeIndex(sorted(pd.Timestamp(d) for d in events.values()))
    pos = dates.searchsorted(index)  # first event >= each date
    pos = np.clip(pos, 0, len(dates) - 1)
    nxt = dates[pos]
    out = (nxt - index).days.astype("float64")
    # dates beyond the last tabulated event have no defined "next"
    out = pd.Series(out, index=index)
    out[index > dates[-1]] = np.nan
    return out


def add_calendar_features(
    index: pd.DatetimeIndex,
    stocking_lead: tuple[int, int] = (14, 56),   # Gulf stocking: 2–8 weeks ahead
    diwali_lead: tuple[int, int] = (7, 35),      # domestic buying: 1–5 weeks ahead
) -> pd.DataFrame:
    """Feature block for a given trading calendar. All ex-ante knowable."""
    df = pd.DataFrame(index=index)
    df.index.name = "date"

    d_ram = _days_to_next(index, RAMADAN_START)
    df["days_to_ramadan"] = d_ram
    df["ramadan_stocking"] = (
        d_ram.between(stocking_lead[0], stocking_lead[1]).astype(float)
    )
    # smooth proximity ramp (0 far away -> 1 at start), 90-day horizon:
    # gives the model a graded signal instead of a step
    df["ramadan_proximity"] = (1.0 - d_ram.clip(0, 90) / 90.0).where(d_ram.notna())

    d_diw = _days_to_next(index, DIWALI)
    df["days_to_diwali"] = d_diw
    df["diwali_window"] = d_diw.between(diwali_lead[0], diwali_lead[1]).astype(float)

    return df
