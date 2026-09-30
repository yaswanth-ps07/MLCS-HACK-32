import streamlit as st
import pandas as pd
import joblib

from src.predict import load_model, get_risk_level


st.set_page_config(
    page_title="Digital Forensic Evidence Triage",
    page_icon="🔐",
    layout="wide"
)

st.title("🔐 Digital Forensic Evidence Triage")
st.write("ML-based cybersecurity evidence classification and risk prioritization")

st.divider()

uploaded_file = st.file_uploader(
    "Upload a cybersecurity CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Evidence")
    st.dataframe(df.head(10))

    if st.button("Analyze Evidence"):

        with st.spinner("Analyzing evidence..."):

            model, preprocessor = load_model()
            feature_names = preprocessor.get_feature_names_out()

            importance_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": model.feature_importances_
            })

            importance_df = importance_df.sort_values(
                "Importance",
                ascending=False
            ).head(10)

            X = df.drop(
                columns=["id", "attack_cat", "label"],
                errors="ignore"
            )

            X_processed = preprocessor.transform(X)

            predictions = model.predict(X_processed)
            probabilities = model.predict_proba(X_processed)[:, 1]

            result_df = pd.DataFrame({
                "Evidence ID": df["id"].values if "id" in df.columns else range(len(df)),
                 "Prediction": [
                    "Attack" if p == 1 else "Normal"
                    for p in predictions
                ],
                "Attack Probability": probabilities.round(4),
                "Risk Level": [
                    get_risk_level(p)
                    for p in probabilities
                ]
            })

        st.subheader("Analysis Results")
        st.dataframe(result_df)

        st.subheader("🚨 Top 20 Highest-Risk Evidence")

        top_risk = result_df.sort_values(
            "Attack Probability",
            ascending=False
        ).head(20)

        st.dataframe(
            top_risk,
            use_container_width=True
        )
        
        st.subheader("Risk Summary")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "High Risk",
                (result_df["Risk Level"] == "High").sum()
            )

        with col2:
            st.metric(
                "Medium Risk",
                (result_df["Risk Level"] == "Medium").sum()
            )

        with col3:
            st.metric(
                "Low Risk",
                (result_df["Risk Level"] == "Low").sum()
            )

        st.subheader("📊 Risk Distribution")

        risk_counts = result_df["Risk Level"].value_counts()

        st.bar_chart(risk_counts)
        st.subheader("🔍 Top 10 Important Features")

        st.bar_chart(
            importance_df.set_index("Feature")
        )