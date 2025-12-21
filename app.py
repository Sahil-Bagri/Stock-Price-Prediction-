import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

st.set_page_config(
    page_title="Stock Price Movement Predictor",
    layout="centered"
)

# Load models
rf_model = joblib.load("random_forest_model.pkl")
xgb_model = joblib.load("xgboost_model.pkl")
scaler = joblib.load("scaler.pkl")
nn_model = load_model("price_movement_nn.h5")

st.title("📈 Stock Price Movement Predictor")
st.write("Predict whether the price will go UP or DOWN using Ensemble Learning")

# Inputs
open_price = st.number_input("Open Price", value=150.0)
high_price = st.number_input("High Price", value=155.0)
low_price = st.number_input("Low Price", value=148.0)
close_price = st.number_input("Close Price", value=152.0)
volume = st.number_input("Volume", value=70000000)
ret = st.number_input("Return", value=0.015, format="%.5f")
ma10 = st.number_input("MA 10", value=149.0)
vol_ma10 = st.number_input("Volume MA 10", value=68000000)

if st.button("Predict Price Movement"):
    input_data = np.array([
        open_price, high_price, low_price, close_price,
        volume, ret, ma10, vol_ma10
    ]).reshape(1, -1)

    rf_prob = rf_model.predict_proba(input_data)[0][1]
    xgb_prob = xgb_model.predict_proba(input_data)[0][1]

    scaled_input = scaler.transform(input_data)
    nn_prob = nn_model.predict(scaled_input)[0][0]

    ensemble_prob = (rf_prob + xgb_prob + nn_prob) / 3
    final_pred = "UP 📈" if ensemble_prob > 0.5 else "DOWN 📉"

    st.subheader("🔍 Model Predictions")
    st.write(f"Random Forest Probability: **{rf_prob:.2f}**")
    st.write(f"XGBoost Probability: **{xgb_prob:.2f}**")
    st.write(f"Neural Network Probability: **{nn_prob:.2f}**")

    st.subheader("✅ Ensemble Result")
    st.success(f"Prediction: **{final_pred}**")
    st.info(f"Confidence: **{ensemble_prob*100:.2f}%**")
