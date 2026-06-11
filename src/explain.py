# src/explain.py

import os
import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


MODEL_PATH = "model/xgboost_model.pkl"

FEATURE_PATH = "model/feature_names.pkl"

REPORT_DIR = "reports"

os.makedirs(
    REPORT_DIR,
    exist_ok=True
)


class FraudExplainer:

    def __init__(self):

        self.model = joblib.load(
            MODEL_PATH
        )

        self.feature_names = (
            joblib.load(
                FEATURE_PATH
            )
        )

        self.explainer = (
            shap.TreeExplainer(
                self.model
            )
        )

    def get_shap_values(
        self,
        X
    ):

        shap_values = (
            self.explainer
            .shap_values(X)
        )

        return shap_values

    def create_summary_plot(
        self,
        X
    ):

        shap_values = (
            self.get_shap_values(X)
        )

        plt.figure()

        shap.summary_plot(
            shap_values,
            X,
            show=False
        )

        plt.tight_layout()

        plt.savefig(
            f"{REPORT_DIR}/shap_summary.png",
            bbox_inches="tight"
        )

        plt.close()

        print(
            "SHAP Summary Plot Saved"
        )

    def create_waterfall_plot(
        self,
        X,
        row_index=0
    ):

        shap_values = (
            self.explainer(X)
        )

        plt.figure()

        shap.plots.waterfall(
            shap_values[row_index],
            show=False
        )

        plt.savefig(
            f"{REPORT_DIR}/shap_waterfall.png",
            bbox_inches="tight"
        )

        plt.close()

        print(
            "Waterfall Plot Saved"
        )

    def get_top_features(
        self,
        X,
        row_index=0,
        top_n=10
    ):

        shap_values = (
            self.get_shap_values(X)
        )

        row_values = np.abs(
            shap_values[row_index]
        )

        feature_importance = (
            pd.DataFrame(
                {
                    "feature":
                    X.columns,
                    "impact":
                    row_values
                }
            )
        )

        feature_importance = (
            feature_importance
            .sort_values(
                by="impact",
                ascending=False
            )
        )

        return (
            feature_importance
            .head(top_n)
        )

    def explain_account(
        self,
        X,
        row_index=0
    ):

        top_features = (
            self.get_top_features(
                X,
                row_index=row_index,
                top_n=5
            )
        )

        feature_list = (
            top_features[
                "feature"
            ]
            .tolist()
        )

        explanation = (
            "This account was flagged because "
        )

        explanation += (
            "the model detected unusual "
            "patterns in "
        )

        explanation += (
            ", ".join(
                feature_list
            )
        )

        explanation += (
            ". These features strongly "
            "influenced the risk score "
            "and are commonly associated "
            "with mule-account behaviour."
        )

        return explanation

    def get_feature_contributions(
        self,
        X,
        row_index=0
    ):

        shap_values = (
            self.get_shap_values(X)
        )

        contributions = (
            pd.DataFrame(
                {
                    "feature":
                    X.columns,
                    "shap_value":
                    shap_values[
                        row_index
                    ]
                }
            )
        )

        contributions = (
            contributions
            .sort_values(
                by="shap_value",
                ascending=False
            )
        )

        return contributions

    def save_account_report(
        self,
        X,
        row_index=0
    ):

        report = (
            self.get_feature_contributions(
                X,
                row_index
            )
        )

        report.to_csv(
            f"{REPORT_DIR}/account_explanation.csv",
            index=False
        )

        return report


def generate_global_explanations(
    X
):

    explainer = (
        FraudExplainer()
    )

    explainer.create_summary_plot(X)

    explainer.create_waterfall_plot(
        X,
        row_index=0
    )

    print(
        "Global explanations generated."
    )


if __name__ == "__main__":

    sample_path = (
        "data/transactions.csv"
    )

    if os.path.exists(
        sample_path
    ):

        print(
            "Run explainability through app.py "
            "or after model training."
        )