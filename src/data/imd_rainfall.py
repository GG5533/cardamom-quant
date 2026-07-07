"""IMDRainfallLoader — Idukki cardamom-belt rainfall with anomaly features.

Historical backfill uses IMDLIB (https://imdlib.readthedocs.io), which
downloads IMD Pune's 0.25-degree gridded daily rainfall (1901→recent) and
exposes it as xarray. This loader:

  1. downloads/reads gridded rain for the configured year range;
  2. subsets the Idukki cardamom hill tract bounding box and takes the
     daily area mean;
  3. computes a day-of-year climatology (smoothed) and rainfall anomalies —
     the real counterpart of the synthetic generator's weather-shock driver.

The gridded product is published with a lag, so the most recent months can
be topped up from IMD district data (kept as a separate ``source`` label so
mixed provenance is always visible).

imdlib is imported lazily: parsing cached NetCDF and the anomaly math work
without it, and unit tests exercise exactly that path.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .base import BaseLoader, ValidationError

logger = logging.getLogger(__name__)


class IMDRainfallLoader(BaseLoader):
    SOURCE = "imd_rain_idukki"
    SCHEMA = {
        "rain_mm": "daily area-mean rainfall over the Idukki box, mm",
        "rain_climatology": "smoothed day-of-year expected rainfall, mm",
        "rain_anomaly": "rain_mm - rain_climatology, mm",
        "source": "'imd_gridded' or 'imd_district' (recent top-up)",
    }

    def __init__(
        self,
        raw_dir: Path = config.IMD_RAW_DIR,
        processed_dir: Path = config.PROCESSED_DIR,
        first_year: int = config.IMD_FIRST_YEAR,
        last_year: int | None = None,
        lat: tuple[float, float] = config.IDUKKI_LAT,
        lon: tuple[float, float] = config.IDUKKI_LON,
    ):
        super().__init__(raw_dir, processed_dir)
        self.first_year = first_year
        self.last_year = last_year or pd.Timestamp.today().year - 1
        self.lat, self.lon = lat, lon

    # ----------------------------------------------------------------- fetch
    def fetch(self, force: bool = False) -> int:
        """Download gridded rain via imdlib into raw_dir (idempotent)."""
        import imdlib  # lazy: heavy optional dependency

        marker = self.raw_dir / f"rain_{self.first_year}_{self.last_year}.done"
        if marker.exists() and not force:
            return 0
        imdlib.get_data("rain", self.first_year, self.last_year, fn_format="yearwise",
                        file_dir=str(self.raw_dir))
        marker.touch()
        return self.last_year - self.first_year + 1

    # ----------------------------------------------------------------- parse
    def parse(self) -> pd.DataFrame:
        """Gridded binary -> Idukki daily area-mean -> anomaly features."""
        import imdlib  # lazy

        data = imdlib.open_data(
            "rain", self.first_year, self.last_year, "yearwise", str(self.raw_dir)
        )
        ds = data.get_xarray()
        # IMD uses large negative sentinels for missing ocean/void cells
        ds = ds.where(ds["rain"] > -50.0)
        box = ds.sel(
            lat=slice(self.lat[0], self.lat[1]), lon=slice(self.lon[0], self.lon[1])
        )
        daily = (
            box["rain"].mean(dim=("lat", "lon")).to_dataframe().rename(
                columns={"rain": "rain_mm"}
            )
        )
        daily.index = pd.to_datetime(daily.index)
        daily.index.name = "date"
        return self.add_anomaly(daily[["rain_mm"]], source="imd_gridded")

    # -------------------------------------------------------------- features
    @staticmethod
    def add_anomaly(daily: pd.DataFrame, source: str = "imd_gridded",
                    smooth_days: int = 15) -> pd.DataFrame:
        """Attach day-of-year climatology + anomaly. Pure — unit-testable.

        Climatology is the multi-year mean per day-of-year, circularly
        smoothed with a centred rolling mean so a sparse history doesn't
        produce a jagged seasonal curve.
        """
        df = daily.copy().sort_index()
        doy = df.index.dayofyear.where(~((df.index.month == 2) & (df.index.day == 29)), 59)
        clim = df["rain_mm"].groupby(doy).mean()
        clim = clim.reindex(range(1, 367)).interpolate(limit_direction="both")
        # circular smoothing: pad both ends before rolling
        padded = pd.concat([clim.iloc[-smooth_days:], clim, clim.iloc[:smooth_days]])
        smoothed = padded.rolling(smooth_days, center=True, min_periods=1).mean()
        clim = smoothed.iloc[smooth_days:-smooth_days]
        clim.index = range(1, 367)

        df["rain_climatology"] = clim.reindex(doy).to_numpy()
        df["rain_anomaly"] = df["rain_mm"] - df["rain_climatology"]
        df["source"] = source
        return df

    # -------------------------------------------------------------- validate
    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().validate(df)
        if df["rain_mm"].lt(0).any():
            raise ValidationError("imd: negative rainfall — sentinel leak")
        if df["rain_mm"].gt(1000).any():  # >1m in a day is a parse bug
            raise ValidationError("imd: rainfall above 1000 mm/day")
        monsoon = df[df.index.month.isin([6, 7, 8])]["rain_climatology"].mean()
        dry = df[df.index.month.isin([1, 2])]["rain_climatology"].mean()
        if pd.notna(monsoon) and pd.notna(dry) and monsoon <= dry:
            raise ValidationError("imd: climatology lacks monsoon seasonality")
        return df
