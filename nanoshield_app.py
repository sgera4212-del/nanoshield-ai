# ============================================
# 🚀 NanoShield AI 2.0
# AI-Driven Nanomaterial Risk Screening System
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error

st.set_page_config(page_title="NanoShield AI 2.0", layout="wide")

st.title("🚀 NanoShield AI 2.0")
st.subheader("AI-Powered Nanomaterial Risk Screening System")

uploaded_file = st.file_uploader("Upload Nanotoxicity Dataset (CSV)", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    required_cols = {"Material", "Size_nm", "Concentration_ug_per_mL", "Toxicity"}

    if not required_cols.issubset(df.columns):
        st.error("Dataset must contain: Material, Size_nm, Concentration_ug_per_mL, Toxicity")
        st.stop()

    # ---------------------------
    # Feature Engineering
    # ---------------------------
    df["Inv_Size"] = 1 / df["Size_nm"]
    df["Log_Conc"] = np.log1p(df["Concentration_ug_per_mL"])

    X = df[["Material", "Size_nm", "Concentration_ug_per_mL", "Inv_Size", "Log_Conc"]]
    y = df["Toxicity"]

    # ---------------------------
    # Train-Test Split
    # ---------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["Material"])
        ],
        remainder="passthrough"
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42)
    }

    results = {}

    st.subheader("📊 Model Comparison")

    for name, model in models.items():

        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)

        results[name] = (pipe, r2, mae)

        st.write(f"### {name}")
        st.write(f"R² Score: {round(r2,3)}")
        st.write(f"MAE: {round(mae,3)}")

    # Select best model
    best_model_name = max(results, key=lambda x: results[x][1])
    best_model = results[best_model_name][0]

    st.success(f"🏆 Best Performing Model: {best_model_name}")

    # ---------------------------
    # Prediction Section
    # ---------------------------
    st.subheader("🔬 Design Nanoparticle & Predict Risk")

    material = st.selectbox("Material", df["Material"].unique())
    size = st.slider("Particle Size (nm)", 5.0, 100.0, 50.0)
    concentration = st.slider("Concentration (µg/mL)", 1.0, 100.0, 20.0)

    if st.button("Analyze Risk"):

        input_df = pd.DataFrame({
            "Material": [material],
            "Size_nm": [size],
            "Concentration_ug_per_mL": [concentration],
            "Inv_Size": [1/size],
            "Log_Conc": [np.log1p(concentration)]
        })

        prediction = best_model.predict(input_df)[0]
        prediction = np.clip(prediction, 0, 1)

        st.metric("Predicted Toxicity Score", round(prediction,3))

        # Risk Classification
        if prediction < 0.3:
            st.success("🟢 LOW RISK")
            st.write("Recommended for preliminary development.")
        elif prediction < 0.6:
            st.warning("🟡 MODERATE RISK")
            st.write("Requires controlled experimental validation.")
        else:
            st.error("🔴 HIGH RISK")
            st.write("Not recommended without safety modification.")

        # Visualization
        st.subheader("📈 Actual vs Predicted (Best Model)")

        test_preds = best_model.predict(X_test)

        fig, ax = plt.subplots()
        ax.scatter(y_test, test_preds)
        ax.set_xlabel("Actual Toxicity")
        ax.set_ylabel("Predicted Toxicity")
        ax.set_title("Model Validation")

        st.pyplot(fig)

else:
    st.info("Upload dataset to begin.")

st.markdown("---")
st.caption("NanoShield AI 2.0 | University Science Fiesta Edition")

