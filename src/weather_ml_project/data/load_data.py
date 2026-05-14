from pathlib import Path

import pandas as pd

from weather_ml_project.utils.helpers import normalize_columns


def _read_excel_file(path: Path) -> pd.DataFrame:
    try:
        return pd.read_excel(path, engine="openpyxl")
    except ValueError:
        return pd.read_excel(path, engine="xlrd")


def _normalize_terrain_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "region_id" not in df.columns and "location_id" in df.columns:
        df["region_id"] = df["location_id"]

    if "temperature" in df.columns:
        df["terrain_temperature"] = df["temperature"]
    else:
        for candidate in ["temp", "temp_mean_c", "temperature_c", "tmean"]:
            if candidate in df.columns:
                df["terrain_temperature"] = df[candidate]
                break

    if "precipitation" in df.columns:
        df["terrain_precipitation"] = df["precipitation"]
    else:
        for candidate in ["precip_mm", "precip_mm_jour", "rain", "rain_mm"]:
            if candidate in df.columns:
                df["terrain_precipitation"] = df[candidate]
                break

    if "dewpoint" in df.columns:
        df["terrain_dewpoint"] = df["dewpoint"]
    elif "dewpoint_c" in df.columns:
        df["terrain_dewpoint"] = df["dewpoint_c"]

    if "humidity" in df.columns:
        df["terrain_humidity"] = df["humidity"]
    elif "humidity_pct" in df.columns:
        df["terrain_humidity"] = df["humidity_pct"]

    if "wind_speed" in df.columns:
        df["terrain_wind_speed"] = df["wind_speed"]
    elif "wind_2m_ms" in df.columns:
        df["terrain_wind_speed"] = df["wind_2m_ms"]

    return df


def _aggregate_daily_terrain(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = normalize_columns(df)
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    elif "date" in df.columns:
        df["datetime"] = pd.to_datetime(df["date"], errors="coerce")

    df = df.dropna(subset=["datetime"], how="any")
    df["date"] = df["datetime"].dt.date
    df = _normalize_terrain_columns(df)

    numeric = df.select_dtypes(include="number").columns.tolist()
    agg = {col: "sum" if "precip" in col else "mean" for col in numeric}
    df = df.groupby(["region_id", "date"], dropna=False).agg(agg).reset_index()
    return df


def load_terrain_data(root: Path) -> pd.DataFrame:
    terrain_files = list(root.rglob("*.csv")) + list(root.rglob("*.xlsx")) + list(root.rglob("*.xls"))
    frames = []
    for path in terrain_files:
        try:
            if path.suffix.lower() == ".csv":
                df = pd.read_csv(path)
            else:
                df = _read_excel_file(path)
        except Exception:
            continue
        df = normalize_columns(df)
        df["region_id"] = df.get("region_id", df.get("location_id"))
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    terrain = pd.concat(frames, ignore_index=True, sort=False)
    return _aggregate_daily_terrain(terrain)


def _standardize_era5(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = normalize_columns(df)
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    if "region_id" not in df.columns and "location_id" in df.columns:
        df["region_id"] = df["location_id"]

    if "era5_temperature_2m" in df.columns:
        df["era5_temperature"] = df["era5_temperature_2m"] - 273.15
    if "temperature_2m_c" in df.columns:
        df["era5_temperature"] = df["temperature_2m_c"]
    if "era5_precip_mm" in df.columns:
        df["era5_precipitation"] = df["era5_precip_mm"]
    if "era5_total_precipitation" in df.columns:
        df["era5_precipitation"] = df["era5_total_precipitation"] * 1000
    if "era5_radiation_mjm2" in df.columns:
        df["era5_radiation"] = df["era5_radiation_mjm2"]
    if "era5_radiation_mj_m2" in df.columns:
        df["era5_radiation"] = df["era5_radiation_mj_m2"]
    if "era5_wind_speed_ms" in df.columns:
        df["era5_wind_speed"] = df["era5_wind_speed_ms"]
    if "era5_surface_pressure" in df.columns:
        df["era5_pressure_hpa"] = df["era5_surface_pressure"] / 100
    if "era5_pressure_hpa" in df.columns:
        df["era5_pressure_hpa"] = df["era5_pressure_hpa"]

    df["year_month"] = df["datetime"].dt.to_period("M").astype(str)
    return df


def _standardize_daily_satellite(df: pd.DataFrame, source: str) -> pd.DataFrame:
    df = normalize_columns(df)
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    elif "date" in df.columns:
        df["datetime"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["datetime"], how="any")
    df["date"] = df["datetime"].dt.date
    if "region_id" not in df.columns and "location_id" in df.columns:
        df["region_id"] = df["location_id"]

    if source == "openmeteo":
        if "om_temp_mean_c" in df.columns:
            df["om_temperature"] = df["om_temp_mean_c"]
        if "om_precip_mm" in df.columns:
            df["om_precipitation"] = df["om_precip_mm"]
        if "om_humidity_pct" in df.columns:
            df["om_humidity"] = df["om_humidity_pct"]
        if "om_wind_speed_ms" in df.columns:
            df["om_wind_speed"] = df["om_wind_speed_ms"]
        if "om_radiation_mjm2" in df.columns:
            df["om_radiation"] = df["om_radiation_mjm2"]
    elif source == "nasa":
        if "temp_mean_c" in df.columns:
            df["nasa_temperature"] = df["temp_mean_c"]
        if "precip_mm_jour" in df.columns:
            df["nasa_precipitation"] = df["precip_mm_jour"]
        if "humidity_pct" in df.columns:
            df["nasa_humidity"] = df["humidity_pct"]
        if "wind_2m_ms" in df.columns:
            df["nasa_wind_speed"] = df["wind_2m_ms"]
        if "radiation_mjm2_jour" in df.columns:
            df["nasa_radiation"] = df["radiation_mjm2_jour"]
    df["source"] = source

    numeric = df.select_dtypes(include="number").columns.tolist()
    if df["date"].duplicated().any():
        agg = {col: "sum" if "precip" in col else "mean" for col in numeric}
        df = df.groupby(["region_id", "date"], dropna=False).agg(agg).reset_index()
    return df


def load_satellite_data(openmeteo_path: Path, nasa_path: Path) -> pd.DataFrame:
    openmeteo = pd.read_csv(openmeteo_path)
    nasa = pd.read_csv(nasa_path)

    openmeteo = _standardize_daily_satellite(openmeteo, "openmeteo")
    nasa = _standardize_daily_satellite(nasa, "nasa")
    return pd.concat([openmeteo, nasa], ignore_index=True, sort=False)


def load_era5_monthly(era5_path: Path) -> pd.DataFrame:
    era5 = pd.read_csv(era5_path)
    return _standardize_era5(era5)


def merge_daily_sources(terrain: pd.DataFrame, satellite_daily: pd.DataFrame, era5_monthly: pd.DataFrame) -> pd.DataFrame:
    df = satellite_daily.copy()
    df["year_month"] = pd.to_datetime(df["date"]).dt.to_period("M").astype(str)

    if not terrain.empty:
        df = df.merge(terrain, on=["region_id", "date"], how="inner", suffixes=("", "_terrain"))
    if not era5_monthly.empty:
        era5_cols = [c for c in era5_monthly.columns if c not in ["region_nom", "lat", "lon", "source"]]
        df = df.merge(era5_monthly[era5_cols], on=["region_id", "year_month"], how="left")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"], how="any")
    return df
