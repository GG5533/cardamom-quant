"""GEFS forecast-rain pilot — proves the keyless path before any backfill.

    python scripts/gefs_pilot.py

The forecast-rain trial (pre-registered direction: information timed to
the 5d label window) needs AS-ISSUED historical forecasts. ECMWF's TIGGE
needs an account; NOAA's GEFS v12 reforecast does not: per-variable GRIB2
files on anonymous S3/HTTPS (noaa-gefs-retrospective), daily 00z inits
2000-2019, total precipitation (apcp_sfc), 0.25 deg, leads to 10 days.
The 2020->present leg would come from the operational GEFS archive
(noaa-gefs-bdp-pds) with .idx byte-range subsetting.

This pilot downloads a few sample init days end-to-end, extracts the
Idukki-box 5-day accumulated precip forecast, sanity-checks it against
monsoon climatology, and prints the measured cost of a full backfill
(bytes/init x ~1,300 reforecast-era auction days). No backfill runs here
— the numbers this prints are what the go/no-go decision is made on.
"""
from __future__ import annotations

import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.config import IDUKKI_LAT, IDUKKI_LON  # noqa: E402

BASE = ("https://noaa-gefs-retrospective.s3.amazonaws.com/GEFSv12/reforecast"
        "/{y}/{ymd}00/c00/Days%3A1-10/apcp_sfc_{ymd}00_c00.grib2")
# monsoon, dry season, and the flood build-up
SAMPLE_DATES = ["20180810", "20190115", "20170615"]
LEAD_DAYS = 5


def fetch(ymd: str, dest: Path) -> tuple[float, float]:
    url = BASE.format(y=ymd[:4], ymd=ymd)
    t0 = time.time()
    urllib.request.urlretrieve(url, dest)
    return dest.stat().st_size / 1e6, time.time() - t0


def idukki_5d_forecast(path: Path) -> float:
    """Sum of 3-hourly APCP over leads 0-120h, area-meaned over the box."""
    import xarray as xr

    ds = xr.open_dataset(path, engine="cfgrib", decode_timedelta=True,
                         backend_kwargs={"indexpath": ""})
    da = ds["tp"] if "tp" in ds else ds[list(ds.data_vars)[0]]
    # GEFS longitudes are 0-360
    lon = tuple(x % 360 for x in IDUKKI_LON)
    box = da.sel(latitude=slice(IDUKKI_LAT[1], IDUKKI_LAT[0]),
                 longitude=slice(lon[0], lon[1]))
    steps = box["step"] <= np.timedelta64(LEAD_DAYS, "D")
    # apcp per step is the accumulation over that 3h window
    total = box.sel(step=steps).sum(dim="step").mean().item()
    ds.close()
    return float(total)


def main() -> None:
    tmp = ROOT / "data" / "raw" / "gefs_pilot"
    tmp.mkdir(parents=True, exist_ok=True)
    sizes, times, results = [], [], {}
    for ymd in SAMPLE_DATES:
        dest = tmp / f"apcp_{ymd}.grib2"
        if not dest.exists():
            mb, sec = fetch(ymd, dest)
        else:
            mb, sec = dest.stat().st_size / 1e6, 0.0
        sizes.append(mb)
        times.append(sec)
        mm = idukki_5d_forecast(dest)
        results[ymd] = mm
        print(f"  {ymd}: file {mb:5.1f} MB in {sec:4.1f}s -> "
              f"Idukki 5d forecast {mm:6.1f} mm")

    print("\n=== PILOT VERDICT ===")
    monsoon, dry = results["20180810"], results["20190115"]
    ok = monsoon > 5 * max(dry, 1e-6) and monsoon > 30
    print(f"  seasonality sanity (monsoon >> dry): "
          f"{monsoon:.0f}mm vs {dry:.0f}mm -> {'PASS' if ok else 'FAIL'}")
    per = float(np.mean(sizes))
    n_era = 1300  # auction days 2014-11 -> 2019-12 (reforecast era)
    print(f"  measured cost: {per:.1f} MB/init -> full reforecast-era "
          f"backfill ~= {per * n_era / 1e3:.1f} GB down, stored as a tiny "
          "CSV (files deleted after extraction)")
    print(f"  wall-clock ~= {np.mean([t for t in times if t > 0]) * n_era / 3600:.1f} h "
          "at pilot speed; resumable, disk-guarded runner is the next step")
    print("  2020+ leg: operational GEFS archive with .idx byte-range "
          "subsetting (smaller per-init cost).")


if __name__ == "__main__":
    main()
