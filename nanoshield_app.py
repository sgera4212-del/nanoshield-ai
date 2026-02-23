if uploaded_file:

    # everything inside here must be indented

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
        # ----------------------------
        # Risk Classification
        # ----------------------------
        if prediction < 0.3:
            risk_level = "LOW"
            st.success("🟢 LOW RISK")
            st.write("Recommended for preliminary development.")

            application = """
            ✅ Suggested Application Pathways:
            • Biomedical coatings  
            • Cosmetic formulations  
            • Drug delivery systems  
            • Tissue engineering scaffolds  
            """

        elif prediction < 0.6:
            risk_level = "MODERATE"
            st.warning("🟡 MODERATE RISK")
            st.write("Requires controlled experimental validation.")

            application = """
            ⚠ Suggested Application Pathways:
            • Antimicrobial surface coatings  
            • Wound healing materials  
            • Water purification systems  
            • Controlled therapeutic formulations  
            """

        else:
            risk_level = "HIGH"
            st.error("🔴 HIGH RISK")
            st.write("Not recommended without safety modification.")

            application = """
            🚫 Suggested Application Pathways:
            • Industrial catalysis  
            • Environmental remediation (non-biological)  
            • External coatings with limited exposure  
            • Restricted laboratory research use  
            """

        # ----------------------------
        # Display Application Pathways
        # ----------------------------
        st.subheader("🏥 Recommended Application Pathways")
        st.markdown(application)

        # ----------------------------
        # Visualization
        # ----------------------------
        st.subheader("📈 Actual vs Predicted (Best Model)")

        test_preds = best_model.predict(X_test)

        fig, ax = plt.subplots()
        ax.scatter(y_test, test_preds)
        ax.set_xlabel("Actual Toxicity")
        ax.set_ylabel("Predicted Toxicity")
        ax.set_title("Model Validation")

        st.pyplot(fig)

