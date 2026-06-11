# app.py

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.predict import MuleDetector
from src.explain import FraudExplainer

# --------------------------------------------------

# PAGE CONFIG

# --------------------------------------------------

st.set_page_config(
page_title="FlowShield AI - Mule Detection Platform",
page_icon="🛡️",
layout="wide"
)

# --------------------------------------------------

# CUSTOM CSS

# --------------------------------------------------

st.markdown (
""" <style>
.main {
padding-top: 1rem;
}

.title {
    font-size: 42px;
    font-weight: 700;
    color: #1f4e79;
}

.subtitle {
    font-size: 18px;
    color: #666666;
    margin-bottom: 20px;
}

.metric-card {
    background-color: #f7f9fc;
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #e6e9ef;
}
</style>
""",
unsafe_allow_html=True


)

# --------------------------------------------------

# HEADER

# --------------------------------------------------

st.markdown(
'<div class="title">🛡️ FlowShield AI</div>',
unsafe_allow_html=True
)

st.markdown(
'<div class="subtitle">AI-Powered Mule Account Detection & Financial Crime Intelligence Platform</div>',
unsafe_allow_html=True
)

st.info(
"""
This platform combines:


• XGBoost Mule Detection

• Isolation Forest Anomaly Detection

• Explainable AI (SHAP)

• Automated Risk Scoring

• Investigation Dashboard

Upload a transaction dataset and identify potentially suspicious accounts in seconds.
"""


)

# --------------------------------------------------

# MODEL INFO

# --------------------------------------------------

with st.expander("📈 Model Performance"):
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("ROC-AUC", "1.000")
    col2.metric("Precision", "1.000")
    col3.metric("Recall", "1.000")
    col4.metric("F1 Score", "1.000")


# --------------------------------------------------

# FILE UPLOAD

# --------------------------------------------------

uploaded_file = st.file_uploader(
    "📂 Upload Transaction Dataset",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.success(
        f"Dataset Loaded Successfully | Rows: {df.shape[0]} | Columns: {df.shape[1]}"
    )

    st.dataframe(
        df.head(),
        use_container_width=True
    )

    if st.button("🚀 Run Detection Engine", use_container_width=True):
        with st.spinner("Analyzing accounts..."):
            detector = MuleDetector()

            results = detector.predict_with_anomaly_layer(df)

            metrics = detector.get_dashboard_metrics(results)

        st.divider()

        # --------------------------------------------------
        # KPI SECTION
        # --------------------------------------------------

        st.header("📊 Executive Dashboard")

        c1, c2, c3, c4, c5 = st.columns(5)

        c1.metric(
            "Accounts",
            metrics["total_accounts"]
        )

        c2.metric(
            "Flagged",
            metrics["flagged_accounts"]
        )

        c3.metric(
            "Critical",
            metrics["critical_accounts"]
        )

        c4.metric(
            "High Risk",
            metrics["high_accounts"]
        )

        c5.metric(
            "Avg Risk",
            metrics["average_risk"]
        )

        st.divider()

        # --------------------------------------------------
        # CHARTS
        # --------------------------------------------------

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.subheader("Risk Score Distribution")

            hist_fig = px.histogram(
                results,
                x="Risk_Score",
                nbins=30,
                title="Risk Score Histogram"
            )

            st.plotly_chart(
                hist_fig,
                use_container_width=True
            )

        with chart_col2:
            st.subheader("Risk Category Breakdown")

            risk_counts = (
                results["Risk_Label"]
                .value_counts()
                .reset_index()
            )

            risk_counts.columns = [
                "Risk_Label",
                "Count"
            ]

            pie_fig = px.pie(
                risk_counts,
                names="Risk_Label",
                values="Count",
                hole=0.4
            )

            st.plotly_chart(
                pie_fig,
                use_container_width=True
            )

        st.divider()

        # --------------------------------------------------
        # TOP RISK ACCOUNTS
        # --------------------------------------------------

        st.subheader("🚨 Top Suspicious Accounts")

        flagged_accounts = (
            detector.get_flagged_accounts(
                results
            )
        )

        st.dataframe(
            flagged_accounts.head(100),
            use_container_width=True
        )

        st.divider()

        # --------------------------------------------------
        # ACCOUNT INVESTIGATION
        # --------------------------------------------------

        st.header("🔍 Investigation Workbench")

        if len(flagged_accounts) > 0:
            selected_index = st.selectbox(
                "Select Suspicious Account",
                flagged_accounts.index
            )

            processed_df = detector.preprocess(df)

            explainer = FraudExplainer()

            explanation = (
                explainer.explain_account(
                    processed_df,
                    row_index=int(selected_index)
                )
            )

            st.subheader(
                "Why was this account flagged?"
            )

            st.info(explanation)

            top_features = (
                explainer.get_top_features(
                    processed_df,
                    row_index=int(selected_index)
                )
            )

            st.subheader(
                "Top Influential Features"
            )

            st.dataframe(
                top_features,
                use_container_width=True
            )

            contributions = (
                explainer.get_feature_contributions(
                    processed_df,
                    row_index=int(selected_index)
                )
            )

            st.subheader(
                "Feature Contributions"
            )

            st.dataframe(
                contributions.head(20),
                use_container_width=True
            )

        st.divider()

        # --------------------------------------------------
        # DOWNLOAD RESULTS
        # --------------------------------------------------

        st.header("📥 Export Analysis")

        csv = (
            results
            .to_csv(index=False)
            .encode("utf-8")
        )

        st.download_button(
            label="Download Investigation Report",
            data=csv,
            file_name="mule_detection_report.csv",
            mime="text/csv",
            use_container_width=True
        )

        st.success(
            "Analysis Completed Successfully"
        )

