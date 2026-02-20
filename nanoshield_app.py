# ============================================
# 🧪 NanoShield AI - ML Version
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="NanoShield AI", layout="centered")

st.title("🧪 NanoShield AI (ML-Powered)")
st.write("Machine Learning-Based Nanoparticle Toxicity Prediction")

# --------------------------------------------
# Upload Dataset
# --------------------------------------------
uploaded_file = st.file_uploader("Upload Nanoparticle Dataset (CSV)", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.success("Dataset Loaded Successfully ✅")
    st.dataframe(df.head())

    # Ensure required columns exist
    required_cols = {"Material", "Size", "Concentration", "Toxicity"}

    if not required_cols.issubset(df.columns):
        st.error("Dataset must contain: Material, Size, Concentration, Toxicity")
        st.stop()

    # --------------------------------------------
    # ML MODEL TRAINING
    # --------------------------------------------

    X = df[["Material", "Size", "Concentration"]]
    y = df["Toxicity"]

    # Encode material column
    categorical_features = ["Material"]
    numeric_features = ["Size", "Concentration"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ],
        remainder="passthrough"
    )

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ])

    model.fit(X, y)

    st.success("ML Model Trained Successfully 🤖")

    # --------------------------------------------
    # USER INPUT
    # --------------------------------------------
    st.subheader("🔬 Enter Nanoparticle Parameters")

    material = st.text_input("Material Type", "Gold")
    size = st.number_input("Particle Size (nm)", min_value=1.0, value=50.0)
    concentration = st.number_input("Concentration (µg/mL)", min_value=0.0, value=10.0)

    run = st.button("🚀 Predict Toxicity")

    if run:

        input_data = pd.DataFrame({
            "Material": [material],
            "Size": [size],
            "Concentration": [concentration]
        })

        prediction = model.predict(input_data)[0]

        # Clamp between 0 and 1
        prediction = max(0, min(prediction, 1))

        st.subheader("🧪 Predicted Toxicity Score")
        st.metric("Toxicity Level (0 - 1)", round(prediction, 3))

        # --------------------------------------------
        # Dose-Response Curve
        # --------------------------------------------
        st.subheader("📈 Dose-Response Analysis")

        dose_range = np.linspace(0, concentration * 2, 50)
        response = prediction * (dose_range / (concentration + 1))

        fig1, ax1 = plt.subplots()
        ax1.plot(dose_range, response)
        ax1.set_xlabel("Concentration (µg/mL)")
        ax1.set_ylabel("Toxic Response")
        ax1.set_title("Dose-Response Curve")

        st.pyplot(fig1)

        # --------------------------------------------
        # Feature Influence (Coefficient-Based)
        # --------------------------------------------
        st.subheader("🧠 Feature Influence")

        # Extract regression coefficients
        reg = model.named_steps["regressor"]
        coefficients = reg.coef_

        importance = np.abs(coefficients)
        importance = importance / importance.sum() * 100

        fig2, ax2 = plt.subplots()
        ax2.bar(range(len(importance)), importance)
        ax2.set_ylabel("Influence (%)")
        ax2.set_title("Model Feature Importance")

        st.pyplot(fig2)

else:
    st.info("Upload a dataset to begin.")

st.markdown("---")
st.caption("NanoShield AI ML Prototype © 2026")


