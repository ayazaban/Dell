from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression

# Maps each API variable to its corresponding terrain ground-truth variable
_API_TO_TERRAIN: dict[str, str] = {
    "om_temperature": "terrain_temperature",
    "om_precipitation": "terrain_precipitation",
    "om_humidity": "terrain_humidity",
    "om_wind_speed": "terrain_wind_speed",
    "om_radiation": "terrain_radiation",
    "nasa_temperature": "terrain_temperature",
    "nasa_precipitation": "terrain_precipitation",
    "nasa_humidity": "terrain_humidity",
    "nasa_wind_speed": "terrain_wind_speed",
    "nasa_radiation": "terrain_radiation",
    "era5_temperature": "terrain_temperature",
    "era5_precipitation": "terrain_precipitation",
    "era5_wind_speed": "terrain_wind_speed",
}


def compute_calibration(
    api_df: pd.DataFrame,
    terrain_df: pd.DataFrame,
    source: str,
    output_path: Path,
    time_key: str = "date",
) -> pd.DataFrame:
    """
    Compute linear calibration coefficients per (region_id, variable) pair.
    Uses terrain_df as ground-truth labels: terrain_var = slope * api_var + intercept.
    Saves coefficients to output_path and returns the coefficient DataFrame.
    """
    join_keys = ["region_id", time_key]
    if not all(k in api_df.columns for k in join_keys) or not all(k in terrain_df.columns for k in join_keys):
        print(f"[calibrate] Skipping {source}: missing join columns {join_keys}.")
        return pd.DataFrame()

    api_cols = [c for c in _API_TO_TERRAIN if c in api_df.columns]
    if not api_cols:
        print(f"[calibrate] No matching API columns found for source '{source}'.")
        return pd.DataFrame()

    terrain_cols_needed = list({_API_TO_TERRAIN[c] for c in api_cols if _API_TO_TERRAIN[c] in terrain_df.columns})
    if not terrain_cols_needed:
        print(f"[calibrate] No matching terrain columns for source '{source}'.")
        return pd.DataFrame()

    merged = api_df[join_keys + api_cols].merge(
        terrain_df[join_keys + terrain_cols_needed],
        on=join_keys,
        how="inner",
    )

    if merged.empty:
        print(f"[calibrate] No overlapping data between {source} and terrain.")
        return pd.DataFrame()

    records = []
    for api_col in api_cols:
        terrain_col = _API_TO_TERRAIN[api_col]
        if terrain_col not in merged.columns:
            continue

        for region_id, group in merged.groupby("region_id"):
            valid = group[[api_col, terrain_col]].dropna()
            if len(valid) < 5:
                continue

            X = valid[[api_col]].values
            y = valid[terrain_col].values

            reg = LinearRegression()
            reg.fit(X, y)
            records.append({
                "region_id": region_id,
                "source": source,
                "variable": api_col,
                "slope": reg.coef_[0],
                "intercept": reg.intercept_,
                "r2": round(reg.score(X, y), 4),
                "n_samples": len(valid),
            })

    if not records:
        print(f"[calibrate] No valid (region, variable) pairs for source '{source}'.")
        return pd.DataFrame()

    coef_df = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    coef_df.to_csv(output_path, index=False)
    print(f"[calibrate] {source}: {len(coef_df)} coefficients saved → {output_path.name}")
    return coef_df


def apply_calibration(df: pd.DataFrame, calibration_path: Path) -> pd.DataFrame:
    """Apply linear calibration per (region_id, variable): corrected = slope * raw + intercept."""
    if not calibration_path.exists():
        return df

    coef_df = pd.read_csv(calibration_path)
    df = df.copy()

    for _, row in coef_df.iterrows():
        region_id = row["region_id"]
        variable = row["variable"]
        slope = float(row["slope"])
        intercept = float(row["intercept"])

        if variable not in df.columns:
            continue

        mask = df["region_id"] == region_id
        df.loc[mask, variable] = df.loc[mask, variable] * slope + intercept

    return df
