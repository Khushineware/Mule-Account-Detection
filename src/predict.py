# src/predict.py

import joblib
import numpy as np
import pandas as pd

from src.preprocess import DataPreprocessor
from src.feature_engineering import FeatureEngineer


MODEL_PATH = "model/xgboost_model.pkl"
PREPROCESSOR_PATH = "model/preprocessor.pkl"
ISO_PATH = "model/isolation_forest.pkl"


class MuleDetector:

    def __init__(self):

        self.model = joblib.load(
            MODEL_PATH
        )

        self.preprocessor = (
            DataPreprocessor.load(
                PREPROCESSOR_PATH
            )
        )

        self.isolation_forest = (
            joblib.load(
                ISO_PATH
            )
        )

        self.feature_engineer = (
            FeatureEngineer()
        )

    def preprocess(self, df):

        df = (
            self.preprocessor
            .transform(df)
        )

        df = (
            self.feature_engineer
            .transform(df)
        )

        return df

    def get_risk_label(
        self,
        score
    ):

        if score < 30:
            return "Low"

        elif score < 60:
            return "Medium"

        elif score < 85:
            return "High"

        else:
            return "Critical"

    def predict(self, raw_df):

        processed_df = (
            self.preprocess(raw_df)
        )

        probabilities = (
            self.model
            .predict_proba(
                processed_df
            )[:, 1]
        )

        risk_scores = (
            probabilities * 100
        )

        predictions = (
            probabilities >= 0.50
        ).astype(int)

        results = raw_df.copy()

        results["Risk_Score"] = (
            risk_scores.round(2)
        )

        results["Prediction"] = (
            predictions
        )

        results["Risk_Label"] = (
            results["Risk_Score"]
            .apply(
                self.get_risk_label
            )
        )

        return results

    def predict_with_anomaly_layer(
        self,
        raw_df
    ):

        processed_df = (
            self.preprocess(raw_df)
        )

        probabilities = (
            self.model
            .predict_proba(
                processed_df
            )[:, 1]
        )

        risk_scores = (
            probabilities * 100
        )

        anomaly_scores = (
            self.isolation_forest
            .decision_function(
                processed_df
            )
        )

        anomaly_flags = (
            self.isolation_forest
            .predict(
                processed_df
            )
        )

        results = raw_df.copy()

        results["Risk_Score"] = (
            risk_scores.round(2)
        )

        results["Anomaly_Score"] = (
            anomaly_scores.round(4)
        )

        results["Anomaly_Flag"] = (
            anomaly_flags
        )

        results["Prediction"] = (
            (
                risk_scores >= 50
            ).astype(int)
        )

        results["Risk_Label"] = (
            results["Risk_Score"]
            .apply(
                self.get_risk_label
            )
        )

        return results

    def get_flagged_accounts(
        self,
        prediction_df
    ):

        return prediction_df[
            prediction_df[
                "Prediction"
            ] == 1
        ].sort_values(
            by="Risk_Score",
            ascending=False
        )

    def get_dashboard_metrics(
        self,
        prediction_df
    ):

        total_accounts = len(
            prediction_df
        )

        flagged_accounts = len(
            prediction_df[
                prediction_df[
                    "Prediction"
                ] == 1
            ]
        )

        critical_accounts = len(
            prediction_df[
                prediction_df[
                    "Risk_Label"
                ] == "Critical"
            ]
        )

        high_accounts = len(
            prediction_df[
                prediction_df[
                    "Risk_Label"
                ] == "High"
            ]
        )

        avg_risk = round(
            prediction_df[
                "Risk_Score"
            ].mean(),
            2
        )

        return {
            "total_accounts":
            total_accounts,

            "flagged_accounts":
            flagged_accounts,

            "critical_accounts":
            critical_accounts,

            "high_accounts":
            high_accounts,

            "average_risk":
            avg_risk
        }

    def export_flagged_accounts(
        self,
        prediction_df,
        save_path=
        "data/flagged_accounts.csv"
    ):

        flagged = (
            self.get_flagged_accounts(
                prediction_df
            )
        )

        flagged.to_csv(
            save_path,
            index=False
        )

        return save_path