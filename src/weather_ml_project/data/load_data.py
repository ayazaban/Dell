import unicodedata
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

from weather_ml_project.utils.helpers import normalize_columns


# ── SpreadsheetML helpers ─────────────────────────────────────────────────────

def _ascii(s: str) -> str:
    """Strip accents and lower-case (for fuzzy column matching)."""
    return unicodedata.normalize("NFD", s).encode("ascii", "ignore").decode().lower()


def _is_spreadsheetml(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            h = f.read(80)
        return b"<?xml" in h or b"\xef\xbb\xbf<?" in h
    except Exception:
        return False


def _parse_sml_row(row_el: ET.Element, ns: str):
    """Return {col_index: value} and {col_index: merge_count} dicts."""
    vals: dict  = {}
    merges: dict = {}
    ci = 1
    for cell in row_el.findall(f"{{{ns}}}Cell"):
        idx = cell.get(f"{{{ns}}}Index")
        if idx is not None:
            ci = int(idx)
        data = cell.find(f"{{{ns}}}Data")
        vals[ci] = data.text if data is not None else None
        merge = int(cell.get(f"{{{ns}}}MergeAcross", 0))
        merges[ci] = merge
        ci += 1 + merge
    return vals, merges


def _read_spreadsheetml(path: Path) -> pd.DataFrame:
    NS = "urn:schemas-microsoft-com:office:spreadsheet"
    try:
        tree = ET.parse(str(path))
    except ET.ParseError:
        return pd.DataFrame()
    root = tree.getroot()
    table = root.find(f".//{{{NS}}}Table")
    if table is None:
        return pd.DataFrame()
    rows = table.findall(f"{{{NS}}}Row")
    if len(rows) < 3:
        return pd.DataFrame()

    h1_vals, h1_merges = _parse_sml_row(rows[0], NS)
    h2_vals, _         = _parse_sml_row(rows[1], NS)

    # Expand group headers across merged columns
    group_map: dict = {}
    for ci, val in h1_vals.items():
        if val:
            for off in range(h1_merges.get(ci, 0) + 1):
                group_map[ci + off] = val

    # Build final column names
    max_col = max(max(h1_vals.keys(), default=0), max(h2_vals.keys(), default=0))
    col_names: dict = {}
    for ci in range(1, max_col + 1):
        if ci == 1:
            col_names[ci] = "datetime"
            continue
        grp = _ascii(group_map[ci]) if ci in group_map and group_map[ci] else ""
        sub = str(h2_vals.get(ci, "") or "").strip()
        if grp and sub:
            col_names[ci] = f"{grp}_{sub}"
        elif grp:
            col_names[ci] = grp
        elif sub:
            col_names[ci] = sub
        else:
            col_names[ci] = f"col_{ci}"

    # Parse data rows
    records = []
    for row in rows[2:]:
        v, _ = _parse_sml_row(row, NS)
        if not v or all(x is None for x in v.values()):
            continue
        records.append({col_names.get(ci, f"col_{ci}"): val for ci, val in v.items()})
    return pd.DataFrame(records)


# ── Excel reader (tries SpreadsheetML first) ──────────────────────────────────

def _read_excel_file(path: Path) -> pd.DataFrame:
    if _is_spreadsheetml(path):
        return _read_spreadsheetml(path)
    try:
        return pd.read_excel(path, engine="openpyxl")
    except Exception:
        return pd.read_excel(path, engine="xlrd")


# ── Column normalisation ──────────────────────────────────────────────────────

def _find_col(df: pd.DataFrame, *keywords: str, exclude: str = "") -> str | None:
    """Return first column name whose ASCII form contains ALL keywords
    (and does not contain `exclude`)."""
    for col in df.columns:
        a = _ascii(str(col))
        if all(k in a for k in keywords) and (not exclude or exclude not in a):
            return col
    return None


def _normalize_terrain_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # temperature mean (avoid dewpoint / rosee columns)
    for kws, exc in [
        (["temp"], "ros"),
        (["temp"], "dew"),
    ]:
        cand = _find_col(df, *kws, exclude="ros")
        if cand and "terrain_temperature" not in df.columns:
            # prefer "moy" sub-variant when multiple temp columns exist
            moy = _find_col(df, "temp", "moy", exclude="ros")
            df["terrain_temperature"] = pd.to_numeric(df[moy or cand], errors="coerce")
            break
    # Replace 0.0 with NaN for temperature: 0°C is not a valid air temp in Morocco
    # (sensors report 0 when they have no reading, not actual 0°C measurement)
    if "terrain_temperature" in df.columns:
        df["terrain_temperature"] = df["terrain_temperature"].replace(0.0, np.nan)
    if "terrain_dewpoint" in df.columns:
        df["terrain_dewpoint"] = df["terrain_dewpoint"].replace(0.0, np.nan)

    # precipitation (prefer somme/sum column)
    cand = _find_col(df, "precip")
    if cand and "terrain_precipitation" not in df.columns:
        somme = _find_col(df, "precip", "somme") or _find_col(df, "precip", "sum")
        df["terrain_precipitation"] = pd.to_numeric(df[somme or cand], errors="coerce")

    # dewpoint
    cand = _find_col(df, "ros") or _find_col(df, "dew")
    if cand and "terrain_dewpoint" not in df.columns:
        moy = _find_col(df, "ros", "moy") or _find_col(df, "dew", "moy") or cand
        df["terrain_dewpoint"] = pd.to_numeric(df[moy], errors="coerce")

    # humidity
    cand = _find_col(df, "humid")
    if cand and "terrain_humidity" not in df.columns:
        moy = _find_col(df, "humid", "moy") or cand
        df["terrain_humidity"] = pd.to_numeric(df[moy], errors="coerce")

    # wind speed
    cand = _find_col(df, "vent") or _find_col(df, "wind")
    if cand and "terrain_wind_speed" not in df.columns:
        moy = (_find_col(df, "vent", "moy") or _find_col(df, "wind", "moy")
               or _find_col(df, "wind_2m") or cand)
        df["terrain_wind_speed"] = pd.to_numeric(df[moy], errors="coerce")

    return df


# ── GPS reader ────────────────────────────────────────────────────────────────

def _read_gps(station_dir: Path):
    """Read lat, lon from GPS.txt if present. Returns (lat, lon) or (None, None)."""
    gps_file = station_dir / "GPS.txt"
    if not gps_file.exists():
        return None, None
    try:
        text = gps_file.read_text(encoding="utf-8").strip()
        parts = [p.strip() for p in text.split(",")]
        if len(parts) >= 2:
            return float(parts[0]), float(parts[1])
    except Exception:
        pass
    return None, None


# ── Daily aggregation ─────────────────────────────────────────────────────────

def _aggregate_daily_terrain(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    elif "date" in df.columns:
        df["datetime"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        return pd.DataFrame()

    df = df.dropna(subset=["datetime"])
    df["date"] = df["datetime"].dt.date
    df = _normalize_terrain_columns(df)

    numeric = df.select_dtypes(include="number").columns.tolist()
    agg = {col: "sum" if "precip" in col else "mean" for col in numeric}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        df = df.groupby(["region_id", "date"], dropna=False).agg(agg).reset_index()
    return df


# ── Public loaders ────────────────────────────────────────────────────────────

def load_terrain_data(root: Path) -> pd.DataFrame:
    terrain_files = (
        list(root.rglob("*.csv"))
        + list(root.rglob("*.xlsx"))
        + list(root.rglob("*.xls"))
    )
    frames = []
    loaded = skipped = 0

    for path in terrain_files:
        try:
            if path.suffix.lower() == ".csv":
                df = pd.read_csv(path)
            else:
                df = _read_excel_file(path)
        except Exception:
            skipped += 1
            continue

        if df.empty:
            skipped += 1
            continue

        df = normalize_columns(df)

        # station folder is the parent directory
        station_dir = path.parent
        station_name = station_dir.name

        # region_id from folder name (everything before " • " or full name)
        if "region_id" not in df.columns:
            rid = station_name.split("•")[0].strip() if "•" in station_name else station_name
            df["region_id"] = rid

        # lat/lon from GPS.txt
        lat, lon = _read_gps(station_dir)
        if lat is not None:
            df["lat"] = lat
            df["lon"] = lon

        frames.append(df)
        loaded += 1

    print(f"[terrain] {loaded} fichiers chargés, {skipped} ignorés sur {len(terrain_files)} trouvés.")

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
    return _standardize_era5(pd.read_csv(era5_path))


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    φ1, φ2 = np.radians(lat1), np.radians(lat2)
    Δφ = np.radians(lat2 - lat1)
    Δλ = np.radians(lon2 - lon1)
    a = np.sin(Δφ / 2) ** 2 + np.cos(φ1) * np.cos(φ2) * np.sin(Δλ / 2) ** 2
    return float(2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))


def _remap_terrain_region_ids(terrain: pd.DataFrame, satellite: pd.DataFrame) -> pd.DataFrame:
    """Replace terrain hex station IDs with the nearest satellite region_id using GPS distance."""
    if terrain.empty or "lat" not in terrain.columns or "lon" not in terrain.columns:
        return terrain

    # Build satellite region reference: one row per region
    sat_ref = (
        satellite[["region_id", "lat", "lon"]]
        .dropna(subset=["lat", "lon"])
        .drop_duplicates("region_id")
    )
    if sat_ref.empty:
        return terrain

    # For each unique terrain station, find the nearest satellite region
    terrain_stations = (
        terrain[["region_id", "lat", "lon"]]
        .dropna(subset=["lat", "lon"])
        .drop_duplicates("region_id")
    )

    mapping: dict = {}
    for _, row in terrain_stations.iterrows():
        best_rid = sat_ref.iloc[0]["region_id"]
        best_dist = float("inf")
        for _, srow in sat_ref.iterrows():
            d = _haversine_km(row["lat"], row["lon"], srow["lat"], srow["lon"])
            if d < best_dist:
                best_dist = d
                best_rid = srow["region_id"]
        mapping[row["region_id"]] = best_rid

    terrain = terrain.copy()
    terrain["region_id"] = terrain["region_id"].map(mapping).fillna(terrain["region_id"])
    n_unique = terrain["region_id"].nunique()
    print(f"[remap] {len(mapping)} stations terrain -> {n_unique} regions satellite (ex: {list(mapping.items())[:3]})")
    return terrain


def merge_daily_sources(
    terrain: pd.DataFrame,
    satellite_daily: pd.DataFrame,
    era5_monthly: pd.DataFrame,
) -> pd.DataFrame:
    df = satellite_daily.copy()
    df["year_month"] = pd.to_datetime(df["date"]).dt.to_period("M").astype(str)

    if not terrain.empty:
        terrain = _remap_terrain_region_ids(terrain, satellite_daily)
        # After remapping, aggregate terrain rows that now share the same (region_id, date)
        terrain_num = terrain.select_dtypes(include="number").columns.tolist()
        terrain_agg = {c: "sum" if "precip" in c else "mean" for c in terrain_num if c not in ["lat", "lon"]}
        terrain_agg["lat"] = "first"
        terrain_agg["lon"] = "first"
        terrain = terrain.groupby(["region_id", "date"], dropna=False).agg(terrain_agg).reset_index()
        df = df.merge(terrain, on=["region_id", "date"], how="inner", suffixes=("", "_terrain"))
    if not era5_monthly.empty:
        era5_cols = [c for c in era5_monthly.columns if c not in ["region_nom", "lat", "lon", "source"]]
        df = df.merge(era5_monthly[era5_cols], on=["region_id", "year_month"], how="left")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"], how="any")
    return df
