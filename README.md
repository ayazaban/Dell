# UM6P Weather ML Project

Projet de calibration et prédiction météo pour UM6P, intégrant ERA5, Open-Meteo, NASA POWER et données terrain.

## Objectif
- Calibrer et corriger les données satellite/API à partir de mesures terrain.
- Prédire la température et la précipitation horaires.
- Générer des sorties CSV avec intervalles de confiance.

## Structure
- `src/weather_ml_project/`: code principal
- `notebooks/`: notebooks d'exploration
- `models/`: modèles sauvegardés
- `reports/`: graphiques et résultats

## Installation
```bash
cd C:/Users/Dell/Projects/UM6P_weather_ml_project
poetry install
poetry shell
```

## Exécution
```bash
poetry run python main.py
```

## Notebook Colab
Un notebook prêt à exécuter se trouve dans `notebooks/um6p_weather_pipeline_colab.ipynb`. Il couvre :
- montage Google Drive
- fusion journalière ERA5 / Open-Meteo / NASA POWER / terrain
- feature engineering anti-leakage
- split 60/20/20 + LOLO spatial
- entraînement Random Forest / XGBoost
- SHAP et calibration

## API
Une API FastAPI minimale est disponible dans `api/app.py`.
Lancer avec :
```bash
uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
```

## Importants
- Le pipeline lit les fichiers ERA5/Open-Meteo/NASA POWER et les fichiers terrain Excel.
- Les variables de type méta-données et identifiants sont exclus du training pour éviter le leakage.
- Le remplissage NaN utilise une médiane locale par région/mois, avec fallback par interpolation et 0.
