"""GEFS reforecast-era forecast-rain backfill (2014-11-07 -> 2019-12-31).

    python scripts/gefs_backfill_reforecast.py

T13 forecast-rain trial groundwork. Pilot (gefs_pilot.py) proved the
keyless path and measured ~27.5 MB/init -> ~36 GB total for this era.
This is the "resumable, disk-guarded runner" the pilot said was the next
step: one init at a time, deleted immediately after extraction, so peak
extra disk usage stays ~30 MB regardless of how much has been transferred.

Resumable: dates already present in the output CSV are skipped, so a
killed/interrupted run just picks back up. Per-date fetch failures are
logged and skipped (missing days show up as gaps, not a crashed run) —
NOAA's anonymous S3 occasionally 404s a specific init.

Output: data/raw/climate/gefs_forecast_rain.csv
  date, forecast_5d_precip_mm, source, fetched_at_utc
"""
from __future__ import annotations

import csv
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.config import IDUKKI_LAT, IDUKKI_LON  # noqa: E402

BASE = ("https://noaa-gefs-retrospective.s3.amazonaws.com/GEFSv12/reforecast"
        "/{y}/{ymd}00/c00/Days%3A1-10/apcp_sfc_{ymd}00_c00.grib2")
LEAD_DAYS = 5
ERA_START, ERA_END = "2014-11-07", "2019-12-31"
OUT = ROOT / "data" / "raw" / "climate" / "gefs_forecast_rain.csv"
TMP = ROOT / "data" / "raw" / "gefs_tmp"
FIELDS = ["date", "forecast_5d_precip_mm", "source", "fetched_at_utc"]
SLEEP_S = 0.3  # be a considerate citizen of NOAA's public bucket


def auction_dates() -> list[str]:
    market = pd.read_parquet(ROOT / "data" / "processed" / "market.parquet")
    era = market.loc[ERA_START:ERA_END]
    return [d.strftime("%Y%m%d") for d in era.index]


def already_done() -> set[str]:
    if not OUT.exists():
        return set()
    df = pd.read_csv(OUT, dtype={"date": str})
    return set(df["date"])


def fetch(ymd: str, dest: Path) -> None:
    url = BASE.format(y=ymd[:4], ymd=ymd)
    urllib.request.urlretrieve(url, dest)


def idukki_5d_forecast(path: Path) -> float:
    import xarray as xr

    ds = xr.open_dataset(path, engine="cfgrib", decode_timedelta=True,
                         backend_kwargs={"indexpath": ""})
    da = ds["tp"] if "tp" in ds else ds[list(ds.data_vars)[0]]
    lon = tuple(x % 360 for x in IDUKKI_LON)
    box = da.sel(latitude=slice(IDUKKI_LAT[1], IDUKKI_LAT[0]),
                 longitude=slice(lon[0], lon[1]))
    steps = box["step"] <= np.timedelta64(LEAD_DAYS, "D")
    total = box.sel(step=steps).sum(dim="step").mean().item()
    ds.close()
    return float(total)


def main() -> None:
    TMP.mkdir(parents=True, exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    dates = auction_dates()
    done = already_done()
    todo = [d for d in dates if d not in done]
    print(f"reforecast backfill: {len(dates)} auction days in era, "
          f"{len(done)} already done, {len(todo)} to fetch")

    is_new = not OUT.exists()
    with open(OUT, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if is_new:
            writer.writeheader()
        n_ok = n_fail = 0
        for i, ymd in enumerate(todo, 1):
            dest = TMP / f"apcp_{ymd}.grib2"
            try:
                fetch(ymd, dest)
                mm = idukki_5d_forecast(dest)
                writer.writerow({
                    "date": ymd,
                    "forecast_5d_precip_mm": f"{mm:.3f}",
                    "source": "reforecast",
                    "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
                })
                fh.flush()
                n_ok += 1
            except (urllib.error.URLError, urllib.error.HTTPError, OSError,
                     KeyError, ValueError) as exc:
                print(f"  [{ymd}] FAILED: {exc}")
                n_fail += 1
            finally:
                dest.unlink(missing_ok=True)
            if i % 50 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} ({n_ok} ok, {n_fail} failed)")
            time.sleep(SLEEP_S)

    print(f"done: {n_ok} fetched, {n_fail} failed, wrote {OUT}")
    if TMP.exists() and not any(TMP.iterdir()):
        TMP.rmdir()


if __name__ == "__main__":
    main()
