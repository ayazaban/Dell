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
## Features du modèle

Les modèles XGBoost utilisent 82 variables d'entrée issues de quatre sources 
fusionnées : les **données terrain** (stations physiques UM6P — température, 
précipitations, humidité, vent et leurs décalages temporels à J-1, J-7, J-14 
et J-28), les données **Open-Meteo**, la réanalyse **ERA5** (ECMWF) et 
**NASA POWER**. S'y ajoutent des features temporelles cycliques (sinus/cosinus 
du jour de l'année et du mois) ainsi que des variables dérivées comme l'anomalie 
de température et le déficit de pression de vapeur (VPD).

L'analyse SHAP révèle que la température terrain J-1 explique à elle seule ~69% 
des prédictions de température, et la précipitation terrain J-1 explique ~51% 
des prédictions de précipitations. Les sources satellite (ERA5, NASA, Open-Meteo) 
contribuent collectivement à ~3–4% de l'importance — ce qui confirme que le 
modèle effectue une **calibration locale** à partir des mesures station, plutôt 
qu'une simple interpolation satellite. En l'absence de données station pour une 
nouvelle région, le modèle bascule automatiquement sur ERA5 comme variable cible 
de substitution.

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
# 📊 Métriques d'Évaluation

## Validation Dataset

| Variable        | RMSE        | R² Score | Bias        |
|-----------------|------------|----------|-------------|
| 🌡️ Température | 0.068 °C   | 1.0000   | +0.001 °C   |
| 🌧️ Précipitation | 28.772 mm | 0.4791   | -0.515 mm   |

📌 Nombre de lignes : **9 849**

---

## Test Dataset

| Variable        | RMSE       | R² Score | Bias        |
|-----------------|-----------|----------|-------------|
| 🌡️ Température | 0.287 °C  | 0.9992   | -0.031 °C   |
| 🌧️ Précipitation | 4.896 mm | 0.2204   | -0.178 mm   |

📌 Nombre de lignes : **14 774**
<img width="959" height="498" alt="image" src="https://github.com/user-attachments/assets/e75fe166-c86b-4b5f-ba42-f4f580c4c986" />
<img width="959" height="499" alt="image" src="https://github.com/user-attachments/assets/eba686df-7029-4a94-89d6-a1305a83c4d2" />

## Figures
<img width="599" height="372" alt="image" src="https://github.com/user-attachments/assets/03bc4283-7549-4238-969d-58460a867895" />
<img width="431" height="366" alt="image" src="https://github.com/user-attachments/assets/a002a92d-edc6-465f-8fd9-286625d3927d" />

<img width="599" height="369" alt="image" src="https://github.com/user-attachments/assets/a7beeb1a-2bf4-49b3-acc9-fc99d3b3ac1f" />

