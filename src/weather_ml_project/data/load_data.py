import unicodedata
import xml.etree.ElementTree as ET
from math import atan2, cos, radians, sin, sqrt
from pathlib import Path

import pandas as pd

from weather_ml_project.utils.helpers import normalize_columns


def _ascii(s: str) -> str:
    """Lowercase + strip accents for fuzzy column matching."""
    return "".join(
        c for c in unicodedata.normalize("NFD", s.lower())
        if unicodedata.category(c) != "Mn"
    )


def _find_col(df: pd.DataFrame, *keywords: str) -> str | None:
    """Return the first column name whose ASCII-normalized form contains any keyword."""
    for col in df.columns:
        col_a = _ascii(col)
        for kw in keywords:
            if kw in col_a:
                return col
    return None


def _read_spreadsheetml(path: Path) -> pd.DataFrame:
    """Parse Microsoft SpreadsheetML (.xls saved as XML) files.
    Handles ss:Index attribute for sparse rows (cells with gaps).
    """
    tree = ET.parse(path)
    root = tree.getroot()

    SS = "urn:schemas-microsoft-com:office:spreadsheet"
    ns = {"ss": SS}

    table = root.find(f".//{{{SS}}}Table")
    if table is None:
        return pd.DataFrame()

    def _parse_row(row_el: ET.Element) -> dict:
        """Returns {col_index: value} for a single row (1-based).
        Handles ss:Index (explicit position) and ss:MergeAcross (merged columns).
        """
        cells = {}
        current_idx = 1
        for cell_el in row_el.findall(f"{{{SS}}}Cell"):
            idx_attr = cell_el.get(f"{{{SS}}}Index")
            if idx_attr is not None:
                current_idx = int(idx_attr)
            data_el = cell_el.find(f"{{{SS}}}Data")
            cells[current_idx] = data_el.text if data_el is not None else None
            # MergeAcross="N" means the cell spans N additional columns
            merge = cell_el.get(f"{{{SS}}}MergeAcross")
            current_idx += 1 + (int(merge) if merge else 0)
        return cells

    row_elements = table.findall(f"{{{SS}}}Row")
    if not row_elements:
        return pd.DataFrame()

    # Determine max column count
    parsed_rows = [_parse_row(r) for r in row_elements]
    max_col = max((max(r.keys()) for r in parsed_rows if r), default=0)

    # Build 2-D list: fill gaps with None
    matrix = []
    for cells in parsed_rows:
        matrix.append([cells.get(i) for i in range(1, max_col + 1)])

    # First row = headers
    headers = [str(h) if h is not None else f"col_{i}" for i, h in enumerate(matrix[0])]
    return pd.DataFrame(matrix[1:], columns=headers)


def _read_excel_file(path: Path) -> pd.DataFrame:
    """Try openpyxl, then xlrd, then SpreadsheetML XML parser."""
    raw = path.read_bytes()[:8]
    # SpreadsheetML starts with UTF-8 BOM + XML declaration
    if raw.startswith(b"\xef\xbb\xbf<?") or raw.startswith(b"<?xml"):
        return _read_spreadsheetml(path)
    try:
        return pd.read_excel(path, engine="openpyxl")
    except Exception:
        return pd.read_excel(path, engine="xlrd")


def _normalize_terrain_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # col_0 is the datetime column in SpreadsheetML exports
    if "col_0" in df.columns and "datetime" not in df.columns:
        df = df.rename(columns={"col_0": "datetime"})

    if "region_id" not in df.columns and "location_id" in df.columns:
        df["region_id"] = df["location_id"]

    def _set(target: str, *keywords: str) -> None:
        if target in df.columns:
            return
        col = _find_col(df, *keywords)
        if col:
            df[target] = pd.to_numeric(df[col], errors="coerce")

    # Temperature: "température [°C]", "temp_mean_c", "tmean", ...
    _set("terrain_temperature", "temperature", "temp_mean", "tmean")

    # Precipitation: "précipitations [mm]", "precip_mm", "rain", "pluie", "somme"
    _set("terrain_precipitation", "precipitation", "precip", "rain", "pluie")

    # Dewpoint: "point de rosée [°C]", "dewpoint"
    _set("terrain_dewpoint", "rosee", "dewpoint", "dew")

    # Humidity: "humidité relative [%]", "humidity_pct"
    _set("terrain_humidity", "humidite", "humidity", "humid")

    # Wind speed: "vitesse du vent [m/s]", "wind_speed", "wind_2m"
    _set("terrain_wind_speed", "vitesse_du_vent", "wind_speed", "wind_2m", "vent")

    # Radiation: "rayonnement solaire [W/m2]", "radiation"
    _set("terrain_radiation", "rayonnement", "radiation", "solar")

    return df


def _aggregate_daily_terrain(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = normalize_columns(df)
    # col_0 is the datetime column in SpreadsheetML terrain exports
    if "col_0" in df.columns and "datetime" not in df.columns:
        df = df.rename(columns={"col_0": "datetime"})
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    elif "date" in df.columns:
        df["datetime"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        # Try any column whose name contains "date" or "time"
        time_candidates = [c for c in df.columns if "date" in c or "time" in c]
        if time_candidates:
            df["datetime"] = pd.to_datetime(df[time_candidates[0]], errors="coerce")
        else:
            return pd.DataFrame()

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
        # Use parent folder name as region_id if the data has none
        if "region_id" not in df.columns or df["region_id"].isna().all():
            df["region_id"] = path.parent.name
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    terrain = pd.concat(frames, ignore_index=True, sort=False)
    return _aggregate_daily_terrain(terrain)


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    a = sin((lat2 - lat1) / 2) ** 2 + cos(lat1) * cos(lat2) * sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * atan2(sqrt(a), sqrt(1 - a))


def remap_terrain_region_ids(
    terrain: pd.DataFrame,
    api_regions: pd.DataFrame,
    terrain_root: Path,
    max_dist_km: float = 10.0,
) -> pd.DataFrame:
    """
    Replace terrain folder-based region_ids with API region codes (R01, R02 …)
    by matching each terrain station GPS.txt to the nearest API lat/lon.
    Only assigns a match if within max_dist_km.
    """
    # Read GPS.txt files: folder_name -> (lat, lon)
    folder_coords: dict[str, tuple[float, float]] = {}
    for gps_file in terrain_root.rglob("GPS.txt"):
        try:
            text = gps_file.read_text(encoding="utf-8", errors="replace").strip()
            lat, lon = map(float, text.split(","))
            folder_coords[gps_file.parent.name] = (lat, lon)
        except Exception:
            continue

    if not folder_coords or api_regions.empty:
        return terrain

    api_ids = api_regions["region_id"].values
    api_lats = api_regions["lat"].values
    api_lons = api_regions["lon"].values

    # Build folder_name -> API region_id mapping
    folder_to_api: dict[str, str] = {}
    for folder_name, (lat, lon) in folder_coords.items():
        dists = [_haversine_km(lat, lon, la, lo) for la, lo in zip(api_lats, api_lons)]
        best_idx = int(min(range(len(dists)), key=lambda i: dists[i]))
        if dists[best_idx] <= max_dist_km:
            folder_to_api[folder_name] = api_ids[best_idx]

    matched = sum(1 for k in terrain["region_id"] if k in folder_to_api)
    print(f"[terrain] region_id mapping: {len(folder_to_api)} stations matched to API codes "
          f"({matched}/{len(terrain)} rows remapped)")

    terrain = terrain.copy()
    terrain["region_id"] = terrain["region_id"].map(folder_to_api).fillna(terrain["region_id"])
    return terrain


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

    # Always aggregate to daily regardless of whether duplicates exist (handles hourly input)
    numeric = df.select_dtypes(include="number").columns.tolist()
    agg = {col: "sum" if "precip" in col else "mean" for col in numeric}
    group_keys = [k for k in ["region_id", "date"] if k in df.columns]
    if group_keys:
        df = df.groupby(group_keys, dropna=False).agg(agg).reset_index()
    return df


def load_openmeteo(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return _standardize_daily_satellite(df, "openmeteo")


def load_nasa(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return _standardize_daily_satellite(df, "nasa")


def load_satellite_data(openmeteo_path: Path, nasa_path: Path) -> pd.DataFrame:
    openmeteo = load_openmeteo(openmeteo_path)
    nasa = load_nasa(nasa_path)
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
