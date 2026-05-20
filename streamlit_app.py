import math
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR  = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

st.set_page_config(
    page_title="Météo Maroc",
    page_icon="🌤️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(170deg,#1b3a5c 0%,#2a6080 45%,#3d7fa0 100%);
    min-height:100vh;
}
[data-testid="stHeader"]  { background:transparent !important; }
[data-testid="stToolbar"] { right:0.5rem; }
.block-container          { max-width:720px; padding-top:1.2rem; }

h1,h2,h3,p,li,label,
.stMarkdown p             { color:white !important; }

/* ── City header ── */
.city-header { text-align:center; padding:24px 0 12px; }
.city-name   { font-size:2.6rem; font-weight:300; letter-spacing:1px;
               color:white; margin:0; }
.city-coords { font-size:.78rem; opacity:.55; color:white; margin:3px 0 8px; }
.city-temp   { font-size:4.8rem; font-weight:100; color:white; line-height:1; }
.city-cond   { font-size:1rem; opacity:.82; color:white; }
.city-range  { font-size:.88rem; opacity:.58; color:white; margin-top:4px; }

/* ── Glass card ── */
.glass {
    background:rgba(255,255,255,.12);
    backdrop-filter:blur(14px);
    -webkit-backdrop-filter:blur(14px);
    border:1px solid rgba(255,255,255,.18);
    border-radius:16px;
    padding:14px 18px;
    margin:8px 0;
    color:white;
}
.card-label {
    font-size:.68rem; font-weight:700; letter-spacing:2px;
    text-transform:uppercase; opacity:.55; color:white;
    border-bottom:1px solid rgba(255,255,255,.12);
    padding-bottom:7px; margin-bottom:2px;
}

/* ── Forecast row ── */
.fc-row {
    display:flex; align-items:center;
    padding:8px 2px;
    border-bottom:1px solid rgba(255,255,255,.08);
    color:white; font-size:.93rem;
}
.fc-row:last-child  { border-bottom:none; }
.fc-day  { width:72px; font-weight:500; }
.fc-icon { width:36px; text-align:center; font-size:1.2rem; }
.fc-pct  { width:54px; text-align:right; padding-right:8px;
           color:#7ec8e3; font-size:.8rem; }
.fc-lo   { width:32px; text-align:right; opacity:.6; font-size:.9rem; }
.fc-bar  { flex:1; padding:0 10px; }
.fc-hi   { width:32px; font-weight:600; }

/* ── Buttons ── */
.stButton>button {
    background:rgba(255,255,255,.18) !important;
    color:white !important;
    border:1px solid rgba(255,255,255,.3) !important;
    border-radius:22px !important;
    font-size:1rem !important;
}
.stButton>button:hover { background:rgba(255,255,255,.32) !important; }
.stDownloadButton>button {
    background:rgba(255,255,255,.12) !important;
    color:white !important;
    border:1px solid rgba(255,255,255,.25) !important;
    border-radius:22px !important;
    width:100% !important;
}

/* ── Number / Text inputs : conteneur vert ── */
[data-testid="stNumberInput"] [data-baseweb="input"],
[data-testid="stTextInput"]   [data-baseweb="input"] {
    background: #1e5c38 !important;
    border: 1px solid rgba(255,255,255,.25) !important;
    border-radius: 10px !important;
}
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"]   input {
    background: green !important;
    color: white !important;
    caret-color: white !important;
}
[data-testid="stNumberInput"] input::placeholder { color:rgba(255,255,255,.4) !important; }

/* boutons +/- */
[data-testid="stNumberInput"] button {
    background: #2e7d52 !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
}
[data-testid="stNumberInput"] button:hover {
    background: #3a9e68 !important;
}

/* ── Selectbox : champ fermé ── */
[data-testid="stSelectbox"] [data-baseweb="select"] > div {
    background: rgba(255,255,255,.12) !important;
    color: white !important;
    border: 1px solid rgba(255,255,255,.25) !important;
    border-radius: 10px !important;
}
[data-testid="stSelectbox"] [data-baseweb="select"] span { color:white !important; }
[data-testid="stSelectbox"] svg { fill:white !important; }

/* ── Dropdown ouvert : fond vert ── */
[data-baseweb="popover"],
[data-baseweb="menu"],
ul[data-baseweb="menu"] {
    background: #1e5c38 !important;
    border: 1px solid rgba(255,255,255,.2) !important;
    border-radius: 10px !important;
}
li[role="option"] {
    background: #1e5c38 !important;
    color: white !important;
}
li[role="option"]:hover,
li[aria-selected="true"] {
    background: #2e7d52 !important;
    color: white !important;
}
div[data-testid="metric-container"] {
    background:rgba(255,255,255,.1);
    border-radius:10px; padding:8px 12px;
}
div[data-testid="metric-container"] label,
div[data-testid="metric-container"] div { color:white !important; }

/* Tab labels white */
.stTabs [data-baseweb="tab"] { color:rgba(255,255,255,.65) !important; }
.stTabs [aria-selected="true"] { color:white !important;
    border-bottom-color:white !important; }
.stTabs [data-baseweb="tab-list"] { background:transparent !important; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_DIR     = ROOT_DIR / "models"
PROCESSED_DIR = ROOT_DIR / "data_processed"
PRED_DIR      = ROOT_DIR / "predictions"

MONTHS_FR = ["Janv.","Févr.","Mars","Avr.","Mai","Juin",
             "Juil.","Août","Sept.","Oct.","Nov.","Déc."]
MONTHS_FULL = ["Janvier","Février","Mars","Avril","Mai","Juin",
               "Juillet","Août","Septembre","Octobre","Novembre","Décembre"]
DAYS_FR = ["Lun.","Mar.","Mer.","Jeu.","Ven.","Sam.","Dim."]

# ── Helpers ───────────────────────────────────────────────────────────────────
def _hav(lat1, lon1, lat2, lon2):
    R = 6371.0
    a = (math.sin(math.radians((lat2-lat1)/2))**2
         + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))
         * math.sin(math.radians((lon2-lon1)/2))**2)
    return R * 2 * math.asin(math.sqrt(a))


def find_nearest(lat, lon, hist):
    lat_col = next((c for c in ["lat","latitude"]  if c in hist.columns), None)
    lon_col = next((c for c in ["lon","longitude"] if c in hist.columns), None)
    nom_col = next((c for c in ["region_nom","region_name","nom"] if c in hist.columns), None)

    if lat_col and lon_col:
        coords = (hist.groupby("region_id")[[lat_col, lon_col]]
                      .mean().reset_index())
        coords["dist"] = coords.apply(
            lambda r: _hav(lat, lon, r[lat_col], r[lon_col]), axis=1)
        best  = coords.loc[coords["dist"].idxmin()]
        rid   = best["region_id"]
        dist  = round(float(best["dist"]), 1)
    else:
        rid  = hist["region_id"].iloc[0]
        dist = None

    if nom_col:
        mask = hist["region_id"] == rid
        name_vals = hist.loc[mask, nom_col].dropna()
        name = str(name_vals.iloc[0]) if not name_vals.empty else rid
    else:
        name = str(rid)
    return rid, name, dist


def _icon(temp, precip):
    if temp < 2  and precip > 0.5: return "❄️"
    if precip > 10:                 return "⛈️"
    if precip > 3:                  return "🌧️"
    if precip > 1:                  return "🌦️"
    if precip > 0.3:                return "🌥️"
    if temp > 32:                   return "🔆"
    if temp > 22:                   return "☀️"
    if temp > 14:                   return "🌤️"
    return "⛅"


def _color(ratio):
    if ratio < 0.25: return "#5fa8d3"
    if ratio < 0.50: return "#78c472"
    if ratio < 0.75: return "#f0c040"
    return "#f07830"


def _amp(month):
    return {1:4,2:5,3:6,4:7,5:7,6:8,7:8,8:7,9:7,10:6,11:5,12:4}.get(month, 5)


def _bar(lo, hi, ylo, yhi, color):
    span = max(yhi - ylo, 1)
    l = max(0.0, min(100.0, (lo - ylo)/span*100))
    r = max(0.0, min(100.0, (hi - ylo)/span*100))
    w = max(r - l, 5)
    return (f'<div style="background:rgba(255,255,255,.18);border-radius:3px;'
            f'height:4px;position:relative;">'
            f'<div style="position:absolute;left:{l:.1f}%;width:{w:.1f}%;'
            f'height:4px;border-radius:3px;background:{color};"></div>'
            f'</div>')


# ── Load models & history ─────────────────────────────────────────────────────
@st.cache_resource
def _load_models():
    """Load all available source models. Returns a unified dict."""
    md: dict = {}

    for src in ["era5", "openmeteo", "nasa", "fusion"]:
        t_path = MODEL_DIR / f"xgb_temperature_{src}.joblib"
        p_path = MODEL_DIR / f"xgb_precipitation_{src}.joblib"
        if not (t_path.exists() and p_path.exists()):
            continue
        try:
            tm = joblib.load(t_path)
            pm = joblib.load(p_path)
            ft_path = MODEL_DIR / f"feature_cols_temp_{src}.joblib"
            fp_path = MODEL_DIR / f"feature_cols_precip_{src}.joblib"
            fc_t = joblib.load(ft_path) if ft_path.exists() else list(tm.get_booster().feature_names)
            fc_p = joblib.load(fp_path) if fp_path.exists() else list(pm.get_booster().feature_names)
            md[f"temperature_{src}"]         = tm
            md[f"precipitation_{src}"]       = pm
            md[f"feature_cols_temp_{src}"]   = fc_t
            md[f"feature_cols_precip_{src}"] = fc_p
        except Exception:
            pass

    # Backward-compat aliases (prefer fusion, fall back to first found)
    for default_src in ["fusion", "era5", "openmeteo", "nasa"]:
        if f"temperature_{default_src}" in md:
            md["temperature"]         = md[f"temperature_{default_src}"]
            md["precipitation"]       = md[f"precipitation_{default_src}"]
            md["feature_cols_temp"]   = md[f"feature_cols_temp_{default_src}"]
            md["feature_cols_precip"] = md[f"feature_cols_precip_{default_src}"]
            md["feature_cols"]        = md[f"feature_cols_temp_{default_src}"]
            break

    # Legacy fallback: old single-model files before multi-source training
    if "temperature" not in md:
        try:
            tm = joblib.load(MODEL_DIR / "xgb_temperature.joblib")
            pm = joblib.load(MODEL_DIR / "xgb_precipitation.joblib")
            fc_t = list(tm.get_booster().feature_names)
            fc_p = list(pm.get_booster().feature_names)
            md["temperature"]         = tm
            md["precipitation"]       = pm
            md["feature_cols_temp"]   = fc_t
            md["feature_cols_precip"] = fc_p
            md["feature_cols"]        = fc_t
        except Exception:
            pass

    return md


@st.cache_data
def _load_hist():
    p = PROCESSED_DIR / "cleaned_data.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()

models_dict = _load_models()
hist_df     = _load_hist()

# Available source variants (for the selector)
_SOURCES_AVAILABLE = [
    s for s in ["fusion", "era5", "openmeteo", "nasa"]
    if f"temperature_{s}" in models_dict
]
_SOURCE_LABELS = {"fusion": "Fusion (tous)", "era5": "ERA5",
                  "openmeteo": "Open-Meteo", "nasa": "NASA POWER"}

# ── Status banner ─────────────────────────────────────────────────────────────
if not models_dict.get("temperature"):
    st.warning("⚠️ Modèles introuvables dans `models/`. Exécutez `python main.py` d'abord.")
if hist_df.empty:
    st.warning("⚠️ Données historiques absentes dans `data_processed/`. Exécutez `python main.py` d'abord.")

# ── Input form ────────────────────────────────────────────────────────────────
_YEARS = list(range(2019, 2031))

# Seed widget keys from last prediction on first run of the session
if "_lat" not in st.session_state:
    st.session_state["_lat"] = st.session_state.get("_pred_lat", 34.30)
if "_lon" not in st.session_state:
    st.session_state["_lon"] = st.session_state.get("_pred_lon", -5.90)
if "_year_sel" not in st.session_state:
    _sy = st.session_state.get("_pred_year")
    st.session_state["_year_sel"] = _sy if _sy in _YEARS else _YEARS[5]

if "_source_sel" not in st.session_state:
    st.session_state["_source_sel"] = _SOURCES_AVAILABLE[0] if _SOURCES_AVAILABLE else "fusion"

st.markdown('<div class="glass">', unsafe_allow_html=True)
if len(_SOURCES_AVAILABLE) > 1:
    c1, c2, c3, c4, c5 = st.columns([2, 2, 1.2, 1.5, 1.2])
else:
    c1, c2, c3, c4 = st.columns([2, 2, 1.5, 1.5])
    c5 = c4  # reuse slot — source selector won't be rendered

with c1:
    user_lat = st.number_input("Latitude", format="%.4f", step=0.01,
                               help="Latitude en degrés décimaux (ex: 34.30)",
                               key="_lat")
with c2:
    user_lon = st.number_input("Longitude", format="%.4f", step=0.01,
                               help="Longitude en degrés décimaux (ex: -5.90)",
                               key="_lon")
with c3:
    year = st.selectbox("Année", _YEARS, key="_year_sel")
if len(_SOURCES_AVAILABLE) > 1:
    with c4:
        source = st.selectbox(
            "Modèle",
            _SOURCES_AVAILABLE,
            format_func=lambda s: _SOURCE_LABELS[s] if s in _SOURCE_LABELS else s,
            key="_source_sel",
        )
    with c5:
        st.markdown("<br>", unsafe_allow_html=True)
        go = st.button("🔍  Prédire", use_container_width=True)
else:
    source = st.session_state["_source_sel"]
    with c4:
        st.markdown("<br>", unsafe_allow_html=True)
        go = st.button("🔍  Prédire", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# Available regions info
if not hist_df.empty:
    lat_col = next((c for c in ["lat","latitude"]  if c in hist_df.columns), None)
    lon_col = next((c for c in ["lon","longitude"] if c in hist_df.columns), None)
    nom_col = next((c for c in ["region_nom","region_name","nom"] if c in hist_df.columns), None)

    with st.expander("📍 Stations disponibles"):
        grp = hist_df.groupby("region_id")
        info_rows = []
        for rid, g in grp:
            row = {"Station": rid}
            if nom_col:
                vals = g[nom_col].dropna()
                row["Nom"] = str(vals.iloc[0]) if not vals.empty else "—"
            if lat_col and lon_col:
                row["Lat"] = f"{g[lat_col].mean():.4f}"
                row["Lon"] = f"{g[lon_col].mean():.4f}"
            if lat_col and lon_col:
                row["Distance (km)"] = round(_hav(user_lat, user_lon, g[lat_col].mean(), g[lon_col].mean()), 1)
            info_rows.append(row)
        if info_rows:
            idf = pd.DataFrame(info_rows)
            if "Distance (km)" in idf.columns:
                idf = idf.sort_values("Distance (km)")
            st.dataframe(idf, use_container_width=True, hide_index=True)

# ── Prediction: run only when button clicked AND inputs differ from cache ──────
if go:
    if not models_dict.get("temperature"):
        st.error("Modèles introuvables. Lancez `python main.py` d'abord.")
        st.stop()
    if hist_df.empty:
        st.error("Données historiques absentes. Lancez `python main.py` d'abord.")
        st.stop()

    _same = (
        "_df" in st.session_state
        and st.session_state.get("_pred_lat") == user_lat
        and st.session_state.get("_pred_lon") == user_lon
        and st.session_state.get("_pred_year") == int(year)
        and st.session_state.get("_pred_source") == source
    )

    if not _same:
        rid, city, dist = find_nearest(user_lat, user_lon, hist_df)

        NEW_LOC_THRESHOLD_KM = 50.0
        is_new_location = dist is not None and dist > NEW_LOC_THRESHOLD_KM

        if is_new_location:
            st.info(
                f"📡 Coordonnées hors des 24 stations connues "
                f"(station la plus proche : {rid}, {dist:.0f} km). "
                f"Téléchargement des données Open-Meteo…"
            )
            prog_bar   = st.progress(0.0)
            status_txt = st.empty()

            def _cb(frac, msg):
                prog_bar.progress(frac)
                status_txt.markdown(f"<small style='color:white'>{msg}</small>",
                                    unsafe_allow_html=True)

            from weather_ml_project.models.predict import predict_new_location
            try:
                df = predict_new_location(
                    models_dict, user_lat, user_lon, int(year), hist_df,
                    source=source, progress_callback=_cb,
                )
                city = f"{user_lat:.3f}°N, {abs(user_lon):.3f}°O"
            except Exception as e:
                st.error(f"Erreur prédiction nouvelle localisation : {e}")
                st.stop()
            finally:
                prog_bar.empty()
                status_txt.empty()
        else:
            if dist and dist > 0:
                st.caption(f"📍 Station la plus proche : **{city}** ({rid}) — {dist:.0f} km")
            with st.spinner(f"Calcul des prédictions pour {city} ({year})…"):
                from weather_ml_project.models.predict import predict_year_for_region
                try:
                    df = predict_year_for_region(models_dict, rid, int(year), hist_df,
                                                 source=source)
                except Exception as e:
                    st.error(f"Erreur lors de la prédiction : {e}")
                    st.stop()

        df["date"]  = pd.to_datetime(df["date"])
        df["month"] = df["date"].dt.month
        df["dow"]   = df["date"].dt.dayofweek

        PRED_DIR.mkdir(exist_ok=True)
        df.to_csv(PRED_DIR / f"predictions_{rid}_{year}.csv", index=False)

        st.session_state.update(
            _df=df, _rid=rid, _city=city,
            _pred_lat=user_lat, _pred_lon=user_lon, _pred_year=int(year),
            _pred_dist=dist, _pred_source=source,
        )

# ── Display: shown whenever a prediction exists in session state ───────────────
if "_df" in st.session_state:
    df   = st.session_state["_df"]
    rid  = st.session_state["_rid"]
    city = st.session_state["_city"]
    disp_year = st.session_state["_pred_year"]
    disp_lat  = st.session_state["_pred_lat"]
    disp_lon  = st.session_state["_pred_lon"]
    dist      = st.session_state.get("_pred_dist")

    # Ensure derived columns exist (safe across reruns)
    df["date"]  = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month
    df["dow"]   = df["date"].dt.dayofweek

    y_lo     = df["temperature_pred"].min() - 3
    y_hi     = df["temperature_pred"].max() + 3
    ann_mean = df["temperature_pred"].mean()
    ann_lo   = df["temperature_pred"].min()
    ann_hi   = df["temperature_pred"].max()
    ann_prec = df["precipitation_pred"].sum()

    ew         = "O" if disp_lon < 0 else "E"
    coord_line = f"{disp_lat:.4f}°N, {abs(disp_lon):.4f}°{ew}"
    dist_line  = f"  •  Station : {rid}" + (f" ({dist} km)" if dist else "")
    first_icon = _icon(float(df.iloc[0]["temperature_pred"]),
                       float(df.iloc[0]["precipitation_pred"]))

    st.markdown(f"""
<div class="city-header">
  <div class="city-name">{city}</div>
  <div class="city-coords">{coord_line}{dist_line}</div>
  <div class="city-temp">{ann_mean:.0f}°</div>
  <div class="city-cond">{first_icon} &nbsp; Température annuelle moyenne</div>
  <div class="city-range">
    Min {ann_lo:.0f}° &nbsp;·&nbsp; Max {ann_hi:.0f}° &nbsp;·&nbsp;
    Précip. totale {ann_prec:.0f} mm/an
  </div>
</div>""", unsafe_allow_html=True)

    tabs = st.tabs(MONTHS_FR)
    for mi, tab in enumerate(tabs):
        m   = mi + 1
        mdf = df[df["month"] == m]
        if mdf.empty:
            continue
        with tab:
            rows_html = ""
            for _, row in mdf.iterrows():
                t      = float(row["temperature_pred"])
                p      = float(row["precipitation_pred"])
                amp    = _amp(m)
                lo, hi = t - amp / 2, t + amp / 2
                ratio  = (t - ann_lo) / max(ann_hi - ann_lo, 1)
                clr    = _color(ratio)
                icon   = _icon(t, p)
                bhtml  = _bar(lo, hi, y_lo, y_hi, clr)
                day_s  = DAYS_FR[int(row["dow"])]
                date_s = row["date"].strftime("%d")
                pstr   = (f'<span style="color:#7ec8e3">{p:.1f}mm</span>'
                          if p >= 0.2 else "")
                rows_html += (
                    f'<div class="fc-row">'
                    f'<div class="fc-day">{day_s} {date_s}</div>'
                    f'<div class="fc-icon">{icon}</div>'
                    f'<div class="fc-pct">{pstr}</div>'
                    f'<div class="fc-lo">{lo:.0f}°</div>'
                    f'<div class="fc-bar">{bhtml}</div>'
                    f'<div class="fc-hi">{hi:.0f}°</div>'
                    f'</div>'
                )
            st.markdown(
                f'<div class="glass">'
                f'<div class="card-label">📅 {MONTHS_FULL[mi]} {disp_year}</div>'
                f'{rows_html}'
                f'</div>',
                unsafe_allow_html=True,
            )
            ca, cb, cc = st.columns(3)
            with ca:
                st.metric("🌡️ Moy.", f"{mdf['temperature_pred'].mean():.1f} °C")
            with cb:
                st.metric("🌧️ Précip.", f"{mdf['precipitation_pred'].sum():.1f} mm")
            with cc:
                st.metric("☔ Jours pluie", int((mdf["precipitation_pred"] > 1).sum()))

    st.markdown("<br>", unsafe_allow_html=True)
    col_dl, col_ok = st.columns([3, 1])
    with col_dl:
        st.download_button(
            label=f"⬇️  Télécharger CSV — {city} {disp_year} ({len(df)} jours)",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=f"predictions_{rid}_{disp_year}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col_ok:
        st.success(f"✅ Sauvegardé\n`predictions/`")
