from typing import List, Optional

import numpy as np
import pandas as pd

from weather_ml_project.config import METADATA_EXCLUSIONS, TARGET_COLUMNS
from weather_ml_project.utils.helpers import normalize_columns


def clean_source(df: pd.DataFrame, source_name: str = "") -> pd.DataFrame:
    """
    Clean a single data source before merging:
      - Remove duplicates on (region_id, date)
      - Fill missing temperature columns with regional monthly mean
      - Fill missing precipitation columns with 0
      - Print IQR outlier counts per column
    """
    df = df.copy()
    df = normalize_columns(df)

    key_cols = [c for c in ["region_id", "date"] if c in df.columns]
    df = df.drop_duplicates(subset=key_cols) if key_cols else df.drop_duplicates()

    if "date" in df.columns:
        df["_month"] = pd.to_datetime(df["date"], errors="coerce").dt.month
        temp_cols = [c for c in df.columns if "temperature" in c or ("temp" in c and c != "_month")]
        for col in temp_cols:
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue
            group_keys = ["region_id", "_month"] if "region_id" in df.columns else ["_month"]
            df[col] = df.groupby(group_keys)[col].transform(lambda s: s.fillna(s.mean()))
            df[col] = df[col].fillna(df[col].mean())
        df = df.drop(columns=["_month"], errors="ignore")

    precip_cols = [c for c in df.columns if "precip" in c or "rain" in c]
    for col in precip_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(0.0)

    _exclude_outlier_check = {"lat", "lon", "latitude", "longitude", "year", "month", "day", "hour"}
    for col in df.select_dtypes(include=[np.number]).columns:
        if col in _exclude_outlier_check:
            continue
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        n_out = ((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum()
        if n_out > 0:
            print(f"[{source_name}] outliers in '{col}': {n_out}")

    return df


def _ensure_date(df: pd.DataFrame) -> pd.DataFrame:
    if "date" not in df.columns:
        if "datetime" in df.columns:
            df["date"] = pd.to_datetime(df["datetime"]).dt.date
        else:
            raise ValueError("Le DataFrame doit contenir la colonne date ou datetime.")
    return df


def _fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = normalize_columns(df)
    df = _ensure_date(df)

    if "region_id" not in df.columns:
        df["region_id"] = df.get("location_id", "unknown")
    df["region_id"] = df["region_id"].fillna("unknown")
    df["month"] = pd.to_datetime(df["date"]).dt.month

    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric = [c for c in numeric if c not in TARGET_COLUMNS]

    for col in numeric:
        df[col] = df.groupby(["region_id", "month"])[col].transform(
            lambda series: series.fillna(series.mean())
        )
        df[col] = df.groupby("region_id")[col].transform(
            lambda series: series.interpolate(method="linear", limit_direction="both")
        )
        df[col] = df[col].fillna(0)

    return df


def _clip_temperature(df: pd.DataFrame, col: str = "terrain_temperature") -> pd.DataFrame:
    if col not in df.columns:
        return df
    q1 = df[col].quantile(0.01)
    q99 = df[col].quantile(0.99)
    df[col] = df[col].clip(q1, q99)
    return df


def _remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates()
    return df


def _detect_outliers(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    # Les coordonnées / métadonnées ont souvent une distribution "atypique"
    # et n'ont pas de sens d'un point de vue outliers pour la modélisation.
    exclude_cols = set(METADATA_EXCLUSIONS) | {"lat", "lon", "latitude", "longitude"}

    for col in columns:
        if col not in df.columns:
            continue
        if col in exclude_cols:
            continue

        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
        print(f"Outliers in {col}: {outliers.sum()}")
        # For now, just print; can clip or remove later

    return df



def _fill_missing_temperature(df: pd.DataFrame) -> pd.DataFrame:
    temp_cols = [col for col in df.columns if 'temperature' in col.lower()]
    for col in temp_cols:
        df[col] = df.groupby(['region_id', pd.to_datetime(df['date']).dt.month])[col].transform(
            lambda x: x.fillna(x.mean())
        )
    return df


def _fill_missing_precipitation(df: pd.DataFrame) -> pd.DataFrame:
    precip_cols = [col for col in df.columns if 'precip' in col.lower()]
    for col in precip_cols:
        df[col] = df.groupby(['region_id', pd.to_datetime(df['date']).dt.month])[col].transform(
            lambda x: x.fillna(x.sum() / len(x))  # Average sum? Wait, sum for missing might not make sense
        )
    return df


def _drop_high_correlation(df: pd.DataFrame, target: str, threshold: float = 0.97) -> pd.DataFrame:
    numeric = df.select_dtypes(include=[np.number]).copy()
    if target not in numeric.columns:
        return df
    corr = numeric.corr()[target].abs()
    drop_cols = [col for col, value in corr.items() if value >= threshold and col != target]
    return df.drop(columns=drop_cols, errors="ignore")


def calibrate_satellite_data(df: pd.DataFrame, calibration_path: str) -> pd.DataFrame:
    """Calibrate satellite data using a calibration CSV file."""
    try:
        calib_df = pd.read_csv(calibration_path)
        # Assume calib_df has columns: region_id, variable, correction_factor
        for _, row in calib_df.iterrows():
            region = row['region_id']
            var = row['variable']
            factor = row['correction_factor']
            if var in df.columns:
                mask = df['region_id'] == region
                df.loc[mask, var] *= factor
    except FileNotFoundError:
        print(f"Calibration file {calibration_path} not found.")
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoyage et imputation des données journalières avant feature engineering."""
    df = normalize_columns(df)
    df = _remove_duplicates(df)
    df = _detect_outliers(df)
    df = _fill_missing_temperature(df)
    df = _fill_missing_precipitation(df)
    df = _fill_missing_values(df)
    df = _clip_temperature(df)

    for target in TARGET_COLUMNS:
        df = _drop_high_correlation(df, target)

    return df
