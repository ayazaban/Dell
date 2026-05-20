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
from weather_ml_project.reports.calibration_report import run_calibration_report
from weather_ml_project.utils.helpers import ensure_directory


def run_pipeline() -> None:
    ensure_directory(PROCESSED_ROOT)
    ensure_directory(MODEL_OUTPUT_DIR)
    ensure_directory(FIGURES_DIR)

    print("[1/6] Chargement des donnees terrain...", flush=True)
    terrain = load_terrain_data(TERRAIN_ROOT)
    if terrain.empty:
        print("     Aucun fichier terrain detecte. Pipeline satellite uniquement.", flush=True)
    else:
        print(f"     {len(terrain)} lignes journalieres chargees.", flush=True)

    print("[2/6] Chargement des donnees satellite...", flush=True)
    satellite_daily = load_satellite_data(OPENMETEO_PATH, NASA_PATH)
    satellite_daily = calibrate_satellite_data(satellite_daily, CALIBRATION_PATH)
    era5_monthly = load_era5_monthly(ERA5_PATH)
    print(f"     {len(satellite_daily)} enregistrements satellite, {len(era5_monthly)} ERA5.", flush=True)

    print("[3/6] Fusion des sources...", flush=True)
    df = merge_daily_sources(terrain, satellite_daily, era5_monthly)
    print(f"     {len(df)} lignes fusionnees.", flush=True)

    print("[4/6] Nettoyage et feature engineering...", flush=True)
    cleaned_df = clean_dataset(df)
    cleaned_df.to_csv(PROCESSED_ROOT / "cleaned_data.csv", index=False)
    df_features = build_features(cleaned_df)

    train_data, val_data, test_data = train_test_split_chronological(df_features)
    print(f"     Train={len(train_data)}  Val={len(val_data)}  Test={len(test_data)}", flush=True)

    print("[5/6] Entrainement des modeles (Optuna ~5 min)...", flush=True)
    model_artifacts = train_models(train_data, val_data)

    print("[6/7] Evaluation et sauvegarde...", flush=True)
    evaluate_models(model_artifacts, val_data, test_data)
    save_predictions(model_artifacts, test_data, MODEL_OUTPUT_DIR)

    print("[7/7] Rapport de calibration par region...", flush=True)
    run_calibration_report(cleaned_df, PROCESSED_ROOT.parent / "reports", FIGURES_DIR)

    print("\nPipeline termine. Modeles et predictions dans:", flush=True)
    print(f"  {MODEL_OUTPUT_DIR}", flush=True)
    print(f"  {PROCESSED_ROOT}", flush=True)

