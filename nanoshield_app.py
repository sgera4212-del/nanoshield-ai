# ============================================
# 🧪 NanoShield AI - Proper ML Version
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error

st.set_page_config(page_title="NanoShield AI", layout="centered")

st.title("🧪 NanoShield AI (ML Model)")
st.write("Supervised Machine Learning for Nanotoxicity Prediction")

uploaded_file = st.file_uploader("Upload Dataset (CSV)", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)
    st.success("Dataset Loaded Successfully ✅")
    st.dataframe(df.head())

    required_cols = {"Material", "Size_nm", "Concentration_ug_per_mL", "Toxicity"}

    if not required_cols.issubset(df.columns):
        st.error("Dataset must contain: Material, Size_nm, Concentration_ug_per_mL, Toxicity")
        st.stop()

    # ------------------------------
    # Features and Target
    # ------------------------------
    X = df[["Material", "Size_nm", "Concentration_ug_per_mL"]]
    y = df["Toxicity"]

    # ------------------------------
    # Train-Test Split
    # ------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ------------------------------
    # Preprocessing + Model
    # ------------------------------
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["Material"])
        ],
        remainder="passthrough"
    )

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ])

    model.fit(X_train, y_train)

    # ------------------------------
    # Model Evaluation
    # ------------------------------
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    st.subheader("📊 Model Performance")
    st.metric("R² Score", round(r2, 3))
    st.metric("Mean Absolute Error", round(mae, 3))

    # ------------------------------
    # Prediction Section
    # ------------------------------
    st.subheader("🔬 Predict Toxicity")

    material = st.selectbox("Material", df["Material"].unique())
    size = st.number_input("Particle Size (nm)", min_value=1.0, value=50.0)
    concentration = st.number_input("Concentration (µg/mL)", min_value=0.0, value=10.0)

    if st.button("🚀 Predict"):

        input_data = pd.DataFrame({
            "Material": [material],
            "Size_nm": [size],
            "Concentration_ug_per_mL": [concentration]
        })

        prediction = model.predict(input_data)[0]
        prediction = max(0, min(prediction, 1))

        st.subheader("🧪 Predicted Toxicity")
        st.metric("Toxicity (0 - 1)", round(prediction, 3))

        # ------------------------------
        # Actual vs Predicted Graph
        # ------------------------------
        st.subheader("📈 Actual vs Predicted")

        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred)
        ax.set_xlabel("Actual Toxicity")
        ax.set_ylabel("Predicted Toxicity")
        ax.set_title("Model Accuracy Visualization")

        st.pyplot(fig)

else:
    st.info("Upload dataset to start.")

st.markdown("---")
st.caption("NanoShield AI | ML Prototype 2026")

