# ============================================
# 🧪 NanoShield AI - Nanoparticle Toxicity App
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="NanoShield AI", layout="centered")

st.title("🧪 NanoShield AI")
st.write("AI-powered Nanoparticle Toxicity Prediction & Analysis")

# --------------------------------------------
# 📂 Upload Dataset (Optional)
# --------------------------------------------
uploaded_file = st.file_uploader("Upload Nanoparticle Dataset (CSV)", type=["csv"])

toxicity_database = {}

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.success("Dataset Loaded Successfully ✅")
    st.write("Preview of Dataset:")
    st.dataframe(df.head())

    # Auto-generate toxicity database if dataset contains required columns
    required_cols = {"Material", "Toxicity"}
    if required_cols.issubset(df.columns):
        toxicity_database = (
            df.groupby("Material")["Toxicity"]
            .mean()
            .to_dict()
        )
    else:
        st.warning("Dataset must contain 'Material' and 'Toxicity' columns to auto-learn toxicity.")

# --------------------------------------------
# 🎛 User Inputs
# --------------------------------------------
st.subheader("🔬 Input Nanoparticle Parameters")

material = st.text_input("Material Type", "Gold")
size = st.number_input("Particle Size (nm)", min_value=1.0, value=50.0)
concentration = st.number_input("Concentration (µg/mL)", min_value=0.0, value=10.0)

# --------------------------------------------
# 🚀 Run Analysis Button
# --------------------------------------------
run = st.button("🚀 Run Analysis", key="run_analysis_btn")

if run:

    # --------------------------------------------
    # 🧠 Toxicity Calculation (Professional Logic)
    # --------------------------------------------

    # Default fallback toxicity
    default_tox = 0.5

    # Use dataset-learned value if available
    tox = toxicity_database.get(material, default_tox)

    # Adjust toxicity slightly based on size & concentration
    size_factor = (1 / size) * 10 if size > 0 else 0
    conc_factor = concentration * 0.01

    final_toxicity = tox + size_factor + conc_factor

    # Normalize between 0 and 1
    final_toxicity = max(0, min(final_toxicity, 1))

    st.subheader("🧪 Predicted Toxicity Score")
    st.metric("Toxicity Level (0 - 1)", round(final_toxicity, 3))

    # --------------------------------------------
    # 📈 Dose-Response Analysis
    # --------------------------------------------
    st.subheader("📈 Dose-Response Analysis")

    dose_range = np.linspace(0, concentration * 2, 50)
    response = final_toxicity * (dose_range / (concentration + 1))

    fig1, ax1 = plt.subplots()
    ax1.plot(dose_range, response)
    ax1.set_xlabel("Concentration (µg/mL)")
    ax1.set_ylabel("Toxic Response")
    ax1.set_title("Dose-Response Curve")

    st.pyplot(fig1)

    # --------------------------------------------
    # 🧠 Feature Influence Analysis
    # --------------------------------------------
    st.subheader("🧠 Feature Influence Analysis")

    material_impact = tox * 20
    size_impact = (1 / size) * 100 if size > 0 else 0
    conc_impact = concentration

    total_impact = material_impact + size_impact + conc_impact

    if total_impact > 0:
        importance = [
            (material_impact / total_impact) * 100,
            (size_impact / total_impact) * 100,
            (conc_impact / total_impact) * 100
        ]
    else:
        importance = [0, 0, 0]

    features = ["Material Type", "Particle Size", "Concentration"]

    fig2, ax2 = plt.subplots()
    ax2.bar(features, importance)
    ax2.set_ylabel("Influence (%)")
    ax2.set_title("Relative Feature Contribution")

    st.pyplot(fig2)

    # --------------------------------------------
    # 📊 Risk Interpretation
    # --------------------------------------------
    st.subheader("⚠ Risk Interpretation")

    if final_toxicity < 0.3:
        st.success("Low Toxicity Risk 🟢")
    elif final_toxicity < 0.7:
        st.warning("Moderate Toxicity Risk 🟡")
    else:
        st.error("High Toxicity Risk 🔴")

# --------------------------------------------
# Footer
# --------------------------------------------
st.markdown("---")
st.caption("NanoShield AI © 2026 | Research Prototype")
