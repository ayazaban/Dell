from typing import List

import numpy as np
import pandas as pd

from weather_ml_project.config import METADATA_EXCLUSIONS, TARGET_COLUMNS


def _ensure_date(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "date" not in df.columns:
        if "datetime" in df.columns:
            df["date"] = pd.to_datetime(df["datetime"]).dt.date
        else:
            raise ValueError("La colonne date ou datetime est requise pour le feature engineering.")
    return df


def _time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month
    df["dayofyear"] = df["date"].dt.dayofyear
    df["dayofweek"] = df["date"].dt.dayofweek
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["doy_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365)
    df["doy_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365)
    return df


def _lag_features(df: pd.DataFrame, columns: List[str], lags: List[int]) -> pd.DataFrame:
    df = df.sort_values(["region_id", "date"])
    for col in columns:
        if col not in df.columns:
            continue
        for lag in lags:
            df[f"{col}_t-{lag}"] = df.groupby("region_id")[col].shift(lag)
    return df


def _build_amplitude_and_vpd(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "om_temp_max_c" in df.columns and "om_temp_min_c" in df.columns:
        df["om_temp_amplitude"] = df["om_temp_max_c"] - df["om_temp_min_c"]
    if "nasa_temperature" in df.columns and "terrain_temperature" in df.columns:
        df["terrain_temp_anomaly"] = df["terrain_temperature"] - df.groupby("region_id")["terrain_temperature"].transform("mean")
    if "terrain_temperature" in df.columns and "terrain_dewpoint" in df.columns:
        temp = df["terrain_temperature"]
        dew = df["terrain_dewpoint"]
        saturated = 0.6108 * np.exp((17.27 * temp) / (temp + 237.3))
        actual = 0.6108 * np.exp((17.27 * dew) / (dew + 237.3))
        df["terrain_vpd"] = saturated - actual
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = _ensure_date(df)
    df = _time_features(df)

    lag_columns = [
        "terrain_temperature",
        "terrain_precipitation",
        "om_temperature",
        "nasa_temperature",
        "era5_temperature",
    ]
    df = _lag_features(df, lag_columns, [1, 7, 14, 28])
    df = _build_amplitude_and_vpd(df)

    df = df.drop(columns=[c for c in df.columns if c in METADATA_EXCLUSIONS and c not in ["date", "region_id"]], errors="ignore")
    df = df.dropna(subset=[c for c in TARGET_COLUMNS if c in df.columns], how="any")
    return df
