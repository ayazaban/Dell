from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

MODEL_DIR = Path(__file__).resolve().parents[1] / 'models'
MODEL_DIR = MODEL_DIR.resolve()

temp_model = joblib.load(MODEL_DIR / 'xgb_temperature.joblib')
precip_model = joblib.load(MODEL_DIR / 'xgb_precipitation.joblib')
feature_cols = joblib.load(MODEL_DIR / 'feature_cols.joblib')

app = FastAPI(title='UM6P Meteo Calibration API')


class PredictionRequest(BaseModel):
    region_id: str | None = None
    date: str
    era5_temperature: float | None = None
    era5_precipitation: float | None = None
    om_temperature: float | None = None
    om_precipitation: float | None = None
    nasa_temperature: float | None = None
    nasa_precipitation: float | None = None
    latitude: float | None = None
    longitude: float | None = None


@app.get('/')
def health():
    return {'status': 'OK', 'message': 'UM6P Meteo Calibration API'}


@app.post('/predict')
def predict(payload: PredictionRequest):
    data = payload.dict()
    df = pd.DataFrame([data])
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['month'] = df['date'].dt.month
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df = df.reindex(columns=feature_cols, fill_value=0)

    temperature = float(temp_model.predict(df)[0])
    precipitation = float(np.expm1(precip_model.predict(df)[0]))

    return {
        'temperature_prediction': temperature,
        'precipitation_prediction': precipitation,
    }
