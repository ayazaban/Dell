from pathlib import Path

import pandas as pd

from weather_ml_project.config import (
    CALIBRATION_PATH,
    ERA5_PATH,
    FIGURES_DIR,
    MODEL_OUTPUT_DIR,
    NASA_PATH,
    OPENMETEO_PATH,
    PROCESSED_ROOT,
    TERRAIN_ROOT,
)
from weather_ml_project.data.load_data import (
    load_era5_monthly,
    load_satellite_data,
    load_terrain_data,
    merge_daily_sources,
)
from weather_ml_project.data.preprocess import calibrate_satellite_data, clean_dataset
from weather_ml_project.features.build_features import build_features
from weather_ml_project.models.evaluate import evaluate_models
from weather_ml_project.models.predict import save_predictions, train_test_split_chronological
from weather_ml_project.models.train import train_models
from weather_ml_project.utils.helpers import ensure_directory


def run_pipeline() -> None:
    ensure_directory(PROCESSED_ROOT)
    ensure_directory(MODEL_OUTPUT_DIR)
    ensure_directory(FIGURES_DIR)

    terrain = load_terrain_data(TERRAIN_ROOT)
    if terrain.empty:
        print("Aucun fichier terrain détecté. Le pipeline continue avec les sources satellite uniquement.")
    else:
        print(f"Données terrain chargées: {len(terrain)} lignes journalières")

    satellite_daily = load_satellite_data(OPENMETEO_PATH, NASA_PATH)
    satellite_daily = calibrate_satellite_data(satellite_daily, CALIBRATION_PATH)
    era5_monthly = load_era5_monthly(ERA5_PATH)
    print(f"Données satellite journalières chargées: {len(satellite_daily)} enregistrements")
    print(f"Données ERA5 mensuelles chargées: {len(era5_monthly)} enregistrements")

    df = merge_daily_sources(terrain, satellite_daily, era5_monthly)
    print(f"Données fusionnées journalières: {len(df)} lignes")

    cleaned_df = clean_dataset(df)
    cleaned_df.to_csv(PROCESSED_ROOT / "cleaned_data.csv", index=False)
    df_features = build_features(cleaned_df)

    train_data, val_data, test_data = train_test_split_chronological(df_features)
    print("Train / val / test sizes:", len(train_data), len(val_data), len(test_data))

    model_artifacts = train_models(train_data, val_data)

    evaluate_models(model_artifacts, val_data, test_data)
    save_predictions(model_artifacts, test_data, MODEL_OUTPUT_DIR)

    print("Pipeline terminé. Modèles et prédictions enregistrés dans:")
    print(f"- {MODEL_OUTPUT_DIR}")
    print(f"- {PROCESSED_ROOT}")

