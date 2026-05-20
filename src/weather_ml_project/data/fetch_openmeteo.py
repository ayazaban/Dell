"""Fetch historical daily weather from Open-Meteo archive for any lat/lon."""
from __future__ import annotations

import pandas as pd
import requests


ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

_DAILY_VARS = [
    "temperature_2m_mean",
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "relative_humidity_2m_mean",
    "wind_speed_10m_mean",
    "shortwave_radiation_sum",
]

_RENAME = {
    "time":                          "date",
    "temperature_2m_mean":           "om_temperature",
    "temperature_2m_max":            "om_temp_max_c",
    "temperature_2m_min":            "om_temp_min_c",
    "precipitation_sum":             "om_precipitation",
    "relative_humidity_2m_mean":     "om_humidity",
    "wind_speed_10m_mean":           "om_wind_speed",
    "shortwave_radiation_sum":       "om_radiation",
    "dew_point_2m_mean":             "om_dewpoint",
    "vapour_pressure_deficit_mean":  "np_vapor_pressure",
}


def fetch_openmeteo_historical(
    lat: float,
    lon: float,
    start_date: str = "2015-01-01",
    end_date: str   = "2024-12-31",
) -> pd.DataFrame:
    """Return a daily DataFrame with Open-Meteo weather features for (lat, lon)."""
    # Filter to variables the API supports; gracefully skip unavailable ones
    available = list(_DAILY_VARS)
    resp = requests.get(
        ARCHIVE_URL,
        params={
            "latitude":   lat,
            "longitude":  lon,
            "start_date": start_date,
            "end_date":   end_date,
            "daily":      ",".join(available),
            "timezone":   "UTC",
        },
        timeout=60,
    )
    resp.raise_for_status()
    daily = resp.json().get("daily", {})

    df = pd.DataFrame(daily).rename(columns=_RENAME)
    df["date"] = pd.to_datetime(df["date"])

    # Aliases expected by feature pipeline
    if "om_temperature" in df.columns:
        df["om_temp_mean_c"] = df["om_temperature"]
    if "om_precipitation" in df.columns:
        df["om_precip_mm"]   = df["om_precipitation"]
    if "om_humidity" in df.columns:
        df["om_humidity_pct"] = df["om_humidity"]
    if "om_wind_speed" in df.columns:
        df["om_wind_speed_ms"] = df["om_wind_speed"]
    if "om_radiation" in df.columns:
        df["om_radiation_mjm2"] = df["om_radiation"]

    df["lat"] = lat
    df["lon"] = lon

    return df
