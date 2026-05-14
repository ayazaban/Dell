import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sys
from pathlib import Path

# Ensure the src/ folder is on sys.path so local package imports work when running streamlit from the repo root.
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

st.set_page_config(page_title="UM6P Weather ML", layout="wide")

# ============================================================================
# Configuration & Loading
# ============================================================================
MODEL_DIR = Path("models")
TEMP_MODEL_FILE = MODEL_DIR / "xgb_temperature.joblib"
PRECIP_MODEL_FILE = MODEL_DIR / "xgb_precipitation.joblib"

@st.cache_resource
def load_models():
    """Load models - feature names extracted from XGBoost objects"""
    try:
        temp_model = joblib.load(TEMP_MODEL_FILE)
        precip_model = joblib.load(PRECIP_MODEL_FILE)
        
        # Extract feature names directly from the trained models
        # This is the authoritative source of features
        temp_feature_names = temp_model.get_booster().feature_names
        precip_feature_names = precip_model.get_booster().feature_names
        
        return temp_model, precip_model, list(temp_feature_names), list(precip_feature_names)
    except Exception as e:
        st.error(f"❌ Erreur: Impossible de charger les modèles - {e}")
        st.stop()

temp_model, precip_model, temp_features, precip_features = load_models()

# Use temperature features as primary (both models should have same features)
feature_cols = temp_features

# ============================================================================
# UI
# ============================================================================
st.title("🌤️ UM6P Weather ML Prediction App")
st.markdown("Application de prédiction météorologique utilisant des modèles XGBoost.")

# Sidebar
st.sidebar.header("⚙️ Configuration")
prediction_type = st.sidebar.selectbox("Type de prédiction", ["Historique", "Année complète"])
target = None
if prediction_type == "Historique":
    target = st.sidebar.selectbox("Cible de prédiction", ["Température", "Précipitations"])
else:
    region = st.sidebar.selectbox("Région", ["R01", "R02", "R03"])  # Add more regions
    year = st.sidebar.number_input("Année", min_value=2020, max_value=2030, value=2026)
st.sidebar.info(f"📊 **Features attendues:** {len(feature_cols)} colonnes")

# ============================================================================
if prediction_type == "Année complète":
    st.header("Prédiction pour une année complète")
    if st.button("Prédire"):
        # Load historical data
        processed_dir = Path("data_processed")
        cleaned_data_path = processed_dir / "cleaned_data.csv"
        if cleaned_data_path.exists():
            historical_data = pd.read_csv(cleaned_data_path)
            # Load models
            models = {"temperature": temp_model, "precipitation": precip_model, "feature_cols": feature_cols}
            from weather_ml_project.models.predict import predict_year_for_region
            preds = predict_year_for_region(models, region, year, historical_data)
            
            # Display results with coordinates
            st.subheader(f"📍 Région: {region}")
            if preds['latitude'].iloc[0] is not None:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Latitude", f"{preds['latitude'].iloc[0]:.4f}")
                with col2:
                    st.metric("Longitude", f"{preds['longitude'].iloc[0]:.4f}")
                with col3:
                    alt = preds['altitude'].iloc[0]
                    st.metric("Altitude", f"{alt:.0f} m" if alt is not None else "N/A")
            
            # Display predictions table
            st.subheader("📊 Prédictions")
            display_cols = ['date', 'region_id', 'latitude', 'longitude', 'altitude', 'temperature_pred', 'precipitation_pred']
            st.dataframe(preds[display_cols].head(20), use_container_width=True)
            
            # Download button
            st.download_button(
                label="⬇️ Télécharger toutes les prédictions (CSV)",
                data=preds.to_csv(index=False),
                file_name=f"predictions_{region}_{year}.csv",
                mime="text/csv"
            )
        else:
            st.error("Données historiques non trouvées. Veuillez exécuter le pipeline d'abord.")
else:
    # Original code for historical prediction
    st.header("Prédiction sur données historiques")
    # ... existing code ...
# ============================================================================
st.header("📊 Données d'Entrée")

# Create base data with all features = 0
input_data = {col: 0.0 for col in feature_cols}

# Update with example values for relevant features
example_values = {
    'era5_temperature_2m': 25.0,
    'era5_total_precipitation': 0.1,
    'era5_dewpoint_temperature_2m': 15.0,
    'era5_surface_pressure': 101300.0,
    'era5_surface_solar_radiation_downwards': 200.0,
    'era5_u_component_of_wind_10m': 3.0,
    'era5_v_component_of_wind_10m': 4.0,
    'era5_wind_speed_ms': 5.0,
    'era5_radiation_mjm2': 200.0,
    'era5_pressure_hpa': 1013.0,
    'era5_precip_mm': 0.1,
    'chirps_precipitation': 0.1,
    'temperature_2m_c': 25.0,
    'dewpoint_temperature_2m_c': 15.0,
    'om_temp_max_c': 30.0,
    'om_temp_min_c': 20.0,
    'om_temp_mean_c': 25.0,
    'om_precip_mm': 0.1,
    'om_humidity_pct': 60.0,
    'om_wind_speed_ms': 5.0,
    'om_radiation_mjm2': 200.0,
    'np_vapor_pressure': 2.0,
    'temp_mean_c': 25.0,
    'temp_max_c': 30.0,
    'temp_min_c': 20.0,
    'dewpoint_c': 15.0,
    'humidity_pct': 60.0,
    'precip_mm_jour': 0.1,
    'wind_2m_ms': 5.0,
    'wind_10m_ms': 6.0,
    'radiation_mjm2_jour': 200.0,
    'pressure_kpa': 101.3,
    'pressure_hpa': 1013.0,
    'dayofyear': 166,
    'month_sin': np.sin(2 * np.pi * 6 / 12),
    'month_cos': np.cos(2 * np.pi * 6 / 12),
    'hour_sin': np.sin(2 * np.pi * 12 / 24),
    'hour_cos': np.cos(2 * np.pi * 12 / 24),
    'doy_sin': np.sin(2 * np.pi * 166 / 365),
    'doy_cos': np.cos(2 * np.pi * 166 / 365),
    'temperature_t-1': 24.8,
    'temperature_t-3': 24.5,
    'temperature_t-6': 24.0,
    'temperature_t-24': 23.5,
    'precipitation_t-1': 0.0,
    'precipitation_t-3': 0.0,
    'precipitation_t-6': 0.0,
    'precipitation_t-24': 0.1,
    'sat_temperature_mean': 25.0,
    'hour_x_summer': 12.0,
    'hour_x_winter': 0.0,
}

# Merge example values into input_data
input_data.update(example_values)

# Display sample data
with st.expander("👀 Voir toutes les features"):
    st.dataframe(pd.DataFrame([input_data]).T, width='stretch')

# ============================================================================
# Feature Alignment (CRITICAL)
# ============================================================================
# Create DataFrame and align features EXACTLY as training
input_df = pd.DataFrame([input_data])

# Step 1: Add missing columns with 0
for col in feature_cols:
    if col not in input_df.columns:
        input_df[col] = 0.0

# Step 2: Select only training features in exact order
input_df = input_df[feature_cols]

# Step 3: Verify alignment
if input_df.shape[1] != len(feature_cols):
    st.error(f"❌ Mismatch de features: {input_df.shape[1]} vs {len(feature_cols)}")
    st.stop()

# ============================================================================
# Prediction
# ============================================================================
if prediction_type == "Historique":
    st.header("🎯 Prédiction")

    if st.button("▶️ Lancer la Prédiction"):
        try:
            if target == "Température":
                prediction = temp_model.predict(input_df)[0]
                st.success(f"**Température Prédite:** {prediction:.2f} °C")
                st.metric("Température", f"{prediction:.2f} °C", delta="Prédiction XGBoost")
                
            elif target == "Précipitations":
                prediction = precip_model.predict(input_df)[0]
                # Inverse log transformation if applied during training
                prediction = np.expm1(prediction.clip(min=0))
                st.success(f"**Précipitations Prédites:** {prediction:.2f} mm")
                st.metric("Précipitations", f"{prediction:.2f} mm", delta="Prédiction XGBoost")
        except Exception as e:
            st.error(f"❌ Erreur lors de la prédiction: {str(e)}")
            with st.expander("📋 Détails d'erreur"):
                st.code(str(e))
else:
    st.info("La prédiction sur une année complète utilise le bouton « Prédire » ci-dessus.")

# ============================================================================
# Info Panel
# ============================================================================
st.markdown("---")
with st.expander("ℹ️ À propos"):
    st.markdown(f"""
    - **Modèle Température:** XGBoost
    - **Modèle Précipitations:** XGBoost
    - **Nombre de features:** {len(feature_cols)}
    - **Type d'données:** Données harmonisées (ERA5, Open-Meteo, NASA POWER, Terrain)
    - **Région:** Maroc (UM6P)
    """)