# src/feature_engineering.py
TARGET_COLUMN = "F3924"
import pandas as pd
import numpy as np


class FeatureEngineer:

    def __init__(self):
        pass

    def create_missing_value_features(self, df):
        """
        Missing values themselves often indicate risk.
        """

        df["missing_feature_count"] = df.isnull().sum(axis=1)

        return df

    def create_zero_value_features(self, df):
        """
        Sparse accounts often have many zeros.
        """

        numeric_cols = df.select_dtypes(
            include=np.number
        ).columns

        df["zero_feature_count"] = (
            (df[numeric_cols] == 0)
            .sum(axis=1)
        )

        return df

    def create_statistical_features(self, df):

        numeric_cols = [
            c for c in df.columns
            if c != "F3924"
            and pd.api.types.is_numeric_dtype(df[c])
        ]

        df["row_mean"] = (
            df[numeric_cols]
            .mean(axis=1)
        )

        df["row_std"] = (
            df[numeric_cols]
            .std(axis=1)
        )

        df["row_min"] = (
            df[numeric_cols]
            .min(axis=1)
        )

        df["row_max"] = (
            df[numeric_cols]
            .max(axis=1)
        )

        df["row_median"] = (
            df[numeric_cols]
            .median(axis=1)
        )

        return df

    def create_percentile_features(self, df):

        numeric_cols = [
            c for c in df.columns
            if c != "F3924"
            and pd.api.types.is_numeric_dtype(df[c])
        ]

        df["row_q25"] = (
            df[numeric_cols]
            .quantile(
                0.25,
                axis=1
            )
        )

        df["row_q75"] = (
            df[numeric_cols]
            .quantile(
                0.75,
                axis=1
            )
        )

        df["row_iqr"] = (
            df["row_q75"]
            - df["row_q25"]
        )

        return df

    def create_outlier_features(self, df):

        numeric_cols = [
            c for c in df.columns
            if c != "F3924"
            and pd.api.types.is_numeric_dtype(df[c])
        ]

        z_scores = (
            df[numeric_cols]
            - df[numeric_cols].mean()
        ) / (
            df[numeric_cols].std()
            + 1e-9
        )

        df["outlier_feature_count"] = (
            np.abs(z_scores) > 3
        ).sum(axis=1)

        return df

    def create_high_risk_bank_features(self, df):
        """
        Bank identified these as important.
        """

        important_cols = [
            "F115",
            "F321",
            "F527",
            "F531",
            "F670",
            "F1692",
            "F2082",
            "F2122",
            "F2582",
            "F2678",
            "F2737",
            "F2956",
            "F3043",
            "F3836",
            "F3887",
            "F3889",
            "F3891",
            "F3894"
        ]

        existing_cols = [
            c
            for c in important_cols
            if c in df.columns
        ]

        if len(existing_cols) > 0:

            df["important_feature_mean"] = (
                df[existing_cols]
                .mean(axis=1)
            )

            df["important_feature_std"] = (
                df[existing_cols]
                .std(axis=1)
            )

            df["important_feature_sum"] = (
                df[existing_cols]
                .sum(axis=1)
            )

        return df

    def create_ratio_features(self, df):

        ratio_pairs = [
            ("F3836", "F3887"),
            ("F3043", "F2956"),
            ("F2737", "F2678"),
            ("F2122", "F2082")
        ]

        for col1, col2 in ratio_pairs:

            if (
                col1 in df.columns
                and col2 in df.columns
            ):

                df[f"{col1}_{col2}_ratio"] = (
                    df[col1]
                    /
                    (df[col2] + 1)
                )

        return df

    def create_risk_density_feature(self, df):

        numeric_cols = [
            c for c in df.columns
            if c != "F3924"
            and pd.api.types.is_numeric_dtype(df[c])
        ]

        high_values = (
            df[numeric_cols]
            >
            df[numeric_cols].quantile(
                0.95
            )
        )

        df["risk_density_score"] = (
            high_values.sum(axis=1)
        )

        return df

    def transform(self, df):

        df = self.create_missing_value_features(df)

        df = self.create_zero_value_features(df)

        df = self.create_statistical_features(df)

        df = self.create_percentile_features(df)

        df = self.create_outlier_features(df)

        df = self.create_high_risk_bank_features(df)

        df = self.create_ratio_features(df)

        df = self.create_risk_density_feature(df)

        return df