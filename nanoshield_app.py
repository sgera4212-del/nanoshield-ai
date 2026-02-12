import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import time
import numpy as np

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="NanoShield AI",
    page_icon="🛡️",
    layout="wide"
)

# -----------------------------
# Dark Tech Blue Theme Styling
# -----------------------------
st.markdown("""
    <style>
    .stApp {
        background-color: #0E1A2B;
        color: white;
    }
    h1, h2, h3 {
        color: #00BFFF;
    }
    .stButton>button {
        background-color: #007ACC;
        color: white;
        border-radius: 8px;
        height: 3em;
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# Title Section
# -----------------------------
st.title("🛡️ NanoShield AI")
st.subheader("AI-Powered Nanotoxicity Risk Assessment System")

st.write("---")

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("🔬 Input Parameters")

material = st.sidebar.selectbox(
    "Nanoparticle Material",
    ["Silver", "Gold", "Zinc Oxide", "Titanium Dioxide"]
)

size = st.sidebar.slider("Particle Size (nm)", 1, 100, 20)
concentration = st.sidebar.slider("Concentration (mg/L)", 0.1, 100.0, 10.0)

# -----------------------------
# Simple Prediction Logic
# -----------------------------
def predict(material, size, concentration):
    base_toxicity = {
        "Silver": 0.8,
        "Gold": 0.3,
        "Zinc Oxide": 0.6,
        "Titanium Dioxide": 0.4
    }

    toxicity = base_toxicity[material] * (concentration / 10) * (50 / size)

    if toxicity > 2:
        risk = "High Risk"
    elif toxicity > 1:
        risk = "Moderate Risk"
    else:
        risk = "Low Risk"

    return round(toxicity, 2), risk

# -----------------------------
# Run Analysis Button
# -----------------------------
if st.button("🚀 Run Analysis"):

    with st.spinner("NanoShield AI is analyzing nanoparticle exposure patterns..."):
        time.sleep(1.5)
        tox, risk = predict(material, size, concentration)

    st.success("Analysis Complete ✅")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Toxicity Score", tox)

    with col2:
        st.metric("Risk Level", risk)

    st.write("---")

    # -----------------------------
    # Dose-Response Curve
    # -----------------------------
  import matplotlib.pyplot as plt
import streamlit as st

# Example inputs
strength = st.slider("Strength", 0, 100, 50)
toxicity = st.slider("Toxicity", 0, 100, 30)

# Create NEW figure every run
fig, ax = plt.subplots()

# Use current values
materials = ["Strength", "Toxicity"]
values = [strength, toxicity]

ax.bar(materials, values)
ax.set_ylim(0, 100)

st.pyplot(fig)

    # -----------------------------
    # Feature Importance (Demo)
    # -----------------------------
    st.subheader("🧠 Feature Influence")

    features = ["Material Type", "Particle Size", "Concentration"]
    importance = [40, 30, 30]

    fig2, ax2 = plt.subplots()
    ax2.bar(features, importance)
    ax2.set_ylabel("Influence (%)")
    st.pyplot(fig2)

# -----------------------------
# CSV Upload Section
# -----------------------------
st.write("---")
st.subheader("📂 Upload CSV for Bulk Analysis")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.write("Preview of Uploaded Data:")
    st.dataframe(df)

    if st.button("Run Bulk Analysis"):
        st.success("Bulk analysis simulated successfully.")

