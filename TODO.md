# TODO - Fix prédiction “Année complète” température constante

- [ ] 1) Mettre à jour `src/weather_ml_project/models/predict.py` : calculer `clim_features` à partir de `region_data_engineered` (et non `region_data` brut) pour que les colonnes de lags soient présentes.
- [ ] 2) Ajouter un garde-fou : compter les features manquantes vs `feature_cols` pour diagnostiquer rapidement.
- [ ] 3) Relancer `python main.py` (pipeline) pour régénérer les artefacts si nécessaire, puis relancer l’app Streamlit et tester “Année complète”.

