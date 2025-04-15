import streamlit as st
import xgboost as xgb
import pandas as pd
import joblib
import datetime
import os

# Charger le modèle à partir du chemin absolu
# MODEL_PATH = r"D:\MEMORY\Prediction_inondation\Prevision_inondation.pkl"
MODEL_PATH = "Prevision_inondation.pkl"
model = joblib.load(MODEL_PATH)

# Interface utilisateur
st.title("🌧️ Prédiction des Inondations à Ouagadougou")
st.markdown("Veuillez renseigner les valeurs suivantes :")

# === Saisie des variables ===
precipitation = st.number_input("Précipitations (mm)", min_value=0.0)
secteur = st.selectbox("Secteur", options=list(range(1, 56)))
superficie_depotoir = st.number_input("Superficie_depotoir (m²)", min_value=0.0)
longueur_caniveau = st.number_input("Longueur_caniveau (m)", min_value=0.0)
plan_eau = st.radio("Plan_eau", ["Oui", "Non"])
type_sol = st.selectbox("Type_sol", ["Hydromorphe", "Peu evolue"])
relief = st.number_input("Relief (élévation moyenne en mètres)", min_value=0.0)
humidite = st.number_input("Humidite (%)", min_value=0.0, max_value=100.0)

# === Saisie de la date ===
st.markdown("### 📅 Date de la prévision")
annee = st.number_input("Annee", min_value=1980, max_value=2030, step=1, value=2023)
mois = st.number_input("Mois", min_value=1, max_value=12, step=1, value=6)
jour = st.number_input("Jour", min_value=1, max_value=31, step=1, value=15)

# === Encodage interne ===
plan_eau_val = 1 if plan_eau == "Oui" else 0
type_sol_map = {"Peu evolue": 0, "Hydromorphe": 1}
type_sol_val = type_sol_map[type_sol]

# === Création du DataFrame pour prédiction ===
X_input = pd.DataFrame([{
    "Precipitation": precipitation,
    "Secteur": secteur,
    "Superficie_depotoir": superficie_depotoir,
    "Longueur_caniveau": longueur_caniveau,
    "Plan_eau": plan_eau_val,
    "Type_sol": type_sol_val,
    "Relief": relief,
    "Humidite": humidite,
    "Annee": annee,
    "Mois": mois,
    "Jour": jour
}])

# === Prédiction ===
if st.button("Prédire"):
    prediction = model.predict(X_input)[0]
    proba = model.predict_proba(X_input)[0][1]

    if prediction == 1:
        st.error(f"🌊 Risque d'inondation détecté : **{proba:.1%}**")
    else:
        st.success(f"✅ Aucun risque d'inondation détecté : **{proba:.1%}**")
