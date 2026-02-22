import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import os 
# Load the model
model = joblib.load("XGBoost.pkl")
feature_names = [
    "Age", "Max_Diameter", "Tumor_Location", "Multifocality", "PLR","Extrathyroidal_extension","SII"]

# Streamlit user interface
st.title("Predictor of Skip metastasis in papillary thyroid cancer ")

# Age: categorical selection (≥55=1, <55=0)
Age = st.selectbox("Age:", options=[0, 1], format_func=lambda x: 'Age < 55 (0)' if x == 0 else 'Age ≥ 55 (1)')

# Max_Diameter: categorical selection (≥10mm=1, <10mm=0)
Max_Diameter = st.selectbox("Max Diameter:", options=[0, 1], format_func=lambda x: 'Diameter < 10mm (0)' if x == 0 else 'Diameter ≥ 10mm (1)')

# Tumor_Location: categorical selection (1=Upper, 2=Middle, 3=Lower)
Tumor_Location = st.selectbox("Tumor Location:", options=[1, 2, 3], format_func=lambda x: {1:'Upper (1)', 2:'Middle (2)', 3:'Lower (3)'}[x])

# Multifocality categorical selection (1=Yes, 0=No)
Multifocality = st.selectbox("Multifocality:", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')

# Extrathyroidal_extension: categorical selection (1=Yes, 0=No)
Extrathyroidal_extension = st.selectbox("Extrathyroidal_extension:", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')

# PLR: numerical input
PLR = st.number_input("PLR (Platelet-to-Lymphocyte Ratio):", min_value=0.0, max_value=1000.0, value=150.0)

# SII: numerical input
SII = st.number_input("SII (Systemic Immune-Inflammation Index):", min_value=0.0, max_value=10000.0, value=500.0)

# Process inputs and make predictions
feature_values = [Age, Max_Diameter, Tumor_Location, Multifocality,Extrathyroidal_extension,PLR, SII]
features = np.array([feature_values])

if st.button("Predict"):
    # Predict class and probabilities
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]

    # Display prediction results
    st.write(f"**Predicted Class:** {predicted_class}")
    st.write(f"**Prediction Probabilities:** {predicted_proba}")

    # Generate advice based on prediction results
    probability = predicted_proba[predicted_class] * 100

    if predicted_class == 1:
        advice = (
            f"According to our model, you have a high risk of skip metastasis in papillary thyroid cancer. "
            f"The model predicts that your probability of having skip metastasis in papillary thyroid cancer is {probability:.1f}%. "
            "While this is just an estimate, it suggests that you may be at significant risk. "
            "I recommend that you consult a thyroid surgeon as soon as possible for further evaluation and "
            "to ensure you receive an accurate diagnosis and necessary treatment."
        )
    else:
        advice = (
            f"According to our model, you have a low risk of skip metastasis in papillary thyroid cancer. "
            f"The model predicts that your probability of not having skip metastasis in papillary thyroid cancer is {probability:.1f}%. "
            "However, maintaining a healthy lifestyle is still very important. "
            "I recommend regular check-ups to monitor your thyroid health, "
            "and to seek medical advice promptly if you experience any symptoms."
        )

    st.write(advice)

    # Calculate SHAP values and display force plot
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_names))

    shap.force_plot(explainer.expected_value, shap_values[0], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)

    st.image("shap_force_plot.png")
