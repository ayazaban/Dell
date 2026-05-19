from pathlib import Path

import pandas as pd

from weather_ml_project.config import (
    CALIBRATION_ERA5_PATH,
    CALIBRATION_NASA_PATH,
    CALIBRATION_OPENMETEO_PATH,
    CALIBRATION_PATH,
    ERA5_PATH,
    FIGURES_DIR,
    MODEL_OUTPUT_DIR,
    NASA_PATH,
    OPENMETEO_PATH,
    PROCESSED_ROOT,
    TERRAIN_ROOT,
)
from weather_ml_project.data.calibrate import apply_calibration, compute_calibration
from weather_ml_project.data.load_data import (
    load_era5_monthly,
    load_nasa,
    load_openmeteo,
    load_terrain_data,
    merge_daily_sources,
    remap_terrain_region_ids,
)
from weather_ml_project.data.preprocess import (
    calibrate_satellite_data,
    clean_dataset,
    clean_source,
)
from weather_ml_project.features.build_features import build_features
from weather_ml_project.models.evaluate import evaluate_models
from weather_ml_project.models.predict import save_predictions, train_test_split_chronological
from weather_ml_project.models.train import train_models
from weather_ml_project.utils.helpers import ensure_directory


def run_pipeline() -> None:
    ensure_directory(PROCESSED_ROOT)
    ensure_directory(MODEL_OUTPUT_DIR)
    ensure_directory(FIGURES_DIR)

    # ── 1. Terrain (hourly → daily + clean) ──────────────────────────────────
    terrain = load_terrain_data(TERRAIN_ROOT)
    if terrain.empty:
        print("Aucun fichier terrain détecté. Pipeline satellite-only.")
    else:
        terrain = clean_source(terrain, "terrain")
        print(f"[terrain] {len(terrain)} lignes journalières après nettoyage")

    # ── 2. Open-Meteo (hourly → daily + clean) ───────────────────────────────
    openmeteo = load_openmeteo(OPENMETEO_PATH)
    openmeteo = clean_source(openmeteo, "openmeteo")

    # ── Remapping region_id terrain → codes API (R01, R02 …) ─────────────────
    if not terrain.empty and "region_id" in openmeteo.columns:
        api_regions = openmeteo[["region_id", "lat", "lon"]].drop_duplicates()
        terrain = remap_terrain_region_ids(terrain, api_regions, TERRAIN_ROOT)
    print(f"[openmeteo] {len(openmeteo)} lignes journalières après nettoyage")

    # ── 3. NASA POWER (hourly → daily + clean) ───────────────────────────────
    nasa = load_nasa(NASA_PATH)
    nasa = clean_source(nasa, "nasa")
    print(f"[nasa] {len(nasa)} lignes journalières après nettoyage")

    # ── 4. ERA5 monthly ───────────────────────────────────────────────────────
    era5 = load_era5_monthly(ERA5_PATH)
    print(f"[era5] {len(era5)} enregistrements mensuels")

    # ── 5. Calibration des APIs avec données terrain comme labels ─────────────
    if not terrain.empty:
        # Open-Meteo calibration (daily alignment on region_id + date)
        compute_calibration(openmeteo, terrain, "openmeteo", CALIBRATION_OPENMETEO_PATH)
        openmeteo = apply_calibration(openmeteo, CALIBRATION_OPENMETEO_PATH)

        # NASA POWER calibration
        compute_calibration(nasa, terrain, "nasa", CALIBRATION_NASA_PATH)
        nasa = apply_calibration(nasa, CALIBRATION_NASA_PATH)

        # ERA5 calibration — align on monthly averages (region_id + year_month)
        if not era5.empty and "year_month" in era5.columns:
            terrain_monthly = terrain.copy()
            terrain_monthly["year_month"] = (
                pd.to_datetime(terrain_monthly["date"], errors="coerce")
                .dt.to_period("M")
                .astype(str)
            )
            terrain_monthly = (
                terrain_monthly.groupby(["region_id", "year_month"])
                .mean(numeric_only=True)
                .reset_index()
            )
            compute_calibration(
                era5, terrain_monthly, "era5", CALIBRATION_ERA5_PATH, time_key="year_month"
            )
            era5 = apply_calibration(era5, CALIBRATION_ERA5_PATH)

    # ── 6. Calibration satellite additionnelle (CSV correction_factor) ────────
    openmeteo = calibrate_satellite_data(openmeteo, CALIBRATION_PATH)
    nasa = calibrate_satellite_data(nasa, CALIBRATION_PATH)

    # ── 7. Fusion des sources ─────────────────────────────────────────────────
    satellite_daily = pd.concat([openmeteo, nasa], ignore_index=True, sort=False)
    df = merge_daily_sources(terrain, satellite_daily, era5)
    print(f"[merge] {len(df)} lignes journalières fusionnées")

    # ── 8. Nettoyage final du DataFrame fusionné ──────────────────────────────
    cleaned_df = clean_dataset(df)
    csv_out = PROCESSED_ROOT / "cleaned_data.csv"
    try:
        cleaned_df.to_csv(csv_out, index=False)
    except PermissionError:
        csv_out = PROCESSED_ROOT / "cleaned_data_new.csv"
        cleaned_df.to_csv(csv_out, index=False)
        print(f"[clean] ⚠ cleaned_data.csv est ouvert (Excel?). Sauvegardé sous cleaned_data_new.csv")
    print(f"[clean] données nettoyées: {len(cleaned_df)} lignes → {csv_out}")

    # ── 9. Feature engineering ────────────────────────────────────────────────
    df_features = build_features(cleaned_df)

    # ── 10. Split temporel ────────────────────────────────────────────────────
    train_data, val_data, test_data = train_test_split_chronological(df_features)
    print(f"[split] train={len(train_data)} | val={len(val_data)} | test={len(test_data)}")

    # ── 11. Entraînement ──────────────────────────────────────────────────────
    model_artifacts = train_models(train_data, val_data)

    # ── 12. Évaluation & prédictions ──────────────────────────────────────────
    evaluate_models(model_artifacts, val_data, test_data)
    save_predictions(model_artifacts, test_data, MODEL_OUTPUT_DIR)

    print("\nPipeline terminé. Résultats dans:")
    print(f"  Modèles  → {MODEL_OUTPUT_DIR}")
    print(f"  Données  → {PROCESSED_ROOT}")
    print(f"  Figures  → {FIGURES_DIR}")
