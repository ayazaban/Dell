from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "src" / "weather_ml_project" / "data"
RAW_ROOT = DATA_ROOT / "raw"
APIS_ROOT = DATA_ROOT / "APIS"

TERRAIN_ROOT = RAW_ROOT / "Données Météo"
ERA5_PATH = APIS_ROOT / "extraction_gee_20260423_154703.csv"
OPENMETEO_PATH = APIS_ROOT / "extraction_open_meteo_20260503_012319.csv"
NASA_PATH = APIS_ROOT / "extraction_nasa_power_quotidienne_20260503_012720.csv"
CALIBRATION_PATH = APIS_ROOT / "calibration.csv"
PROCESSED_ROOT = REPO_ROOT / "data_processed"

METADATA_EXCLUSIONS = [
    # Temporels (utilisés après)
    "datetime",
    "date",
    "time",
    "year",
    "month",
    "day",
    "hour",
    "week",
    "dayofweek",
    # Spatiaux (utilisés après)
    "latitude",
    "longitude",
    "lat",
    "lon",
    "location_id",
    "station_id",
    "region_id",
    "region_name",
    # Noms de colonnes exactes
    "geometry",
    "geom",
    "shapefile",
    "country",
    "province",
    "station_name",
    "source",
    "region_nom",
]

TARGET_COLUMNS = ["temperature", "precipitation"]

MODEL_OUTPUT_DIR = Path("models")
REPORTS_DIR = Path("reports")
FIGURES_DIR = REPORTS_DIR / "figures"
