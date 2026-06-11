# src/preprocess.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
import os


TARGET_COLUMN = "F3924"


class DataPreprocessor:
    """
    Handles:
    - Missing values
    - -1 sentinel values
    - Date features
    - Categorical encoding
    - Train / inference consistency
    """

    def __init__(self):
        self.label_encoders = {}
        self.feature_columns = None

    def load_data(self, filepath):
        """
        Load CSV file
        """
        return pd.read_csv(filepath)

    def replace_sentinel_values(self, df):
        """
        Replace bank sentinel values.
        In this dataset -1 frequently means missing.
        """

        df = df.replace(-1, np.nan)

        return df

    def create_date_features(self, df):
        """
        Create useful features from F3888.
        """

        if "F3888" not in df.columns:
            return df

        try:
            df["F3888"] = pd.to_datetime(
                df["F3888"],
                errors="coerce"
            )

            df["account_year"] = df["F3888"].dt.year
            df["account_month"] = df["F3888"].dt.month
            df["account_day"] = df["F3888"].dt.day
            df["account_weekday"] = df["F3888"].dt.weekday

            reference_date = pd.Timestamp.today()

            df["account_age_days"] = (
                reference_date - df["F3888"]
            ).dt.days

            df.drop(columns=["F3888"], inplace=True)

        except Exception:
            pass

        return df

    def identify_column_types(self, df):

        categorical_cols = df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        numerical_cols = df.select_dtypes(
            include=[
                np.number,
                "float64",
                "int64"
            ]
        ).columns.tolist()

        return categorical_cols, numerical_cols

    def handle_missing_values(self, df):

        categorical_cols, numerical_cols = (
            self.identify_column_types(df)
        )

        for col in numerical_cols:

            median_value = df[col].median()

            if pd.isna(median_value):
                median_value = 0

            df[col] = df[col].fillna(median_value)

        for col in categorical_cols:

            df[col] = df[col].fillna("Unknown")

        return df

    def encode_categorical(self, df, training=True):

        categorical_cols, _ = (
            self.identify_column_types(df)
        )

        for col in categorical_cols:

            if training:

                encoder = LabelEncoder()

                df[col] = encoder.fit_transform(
                    df[col].astype(str)
                )

                self.label_encoders[col] = encoder

            else:

                if col in self.label_encoders:

                    encoder = self.label_encoders[col]

                    known_classes = set(
                        encoder.classes_
                    )

                    df[col] = df[col].astype(str)

                    df[col] = df[col].apply(
                        lambda x:
                        x if x in known_classes
                        else "Unknown"
                    )

                    if (
                        "Unknown"
                        not in encoder.classes_
                    ):

                        encoder.classes_ = np.append(
                            encoder.classes_,
                            "Unknown"
                        )

                    df[col] = encoder.transform(
                        df[col]
                    )

        return df

    def remove_constant_columns(
        self,
        df
    ):
        """
        Remove columns with no information.
        """

        constant_cols = [
            col
            for col in df.columns
            if df[col].nunique() <= 1
        ]

        df = df.drop(
            columns=constant_cols,
            errors="ignore"
        )

        return df

    def fit_transform(self, df):

        df = self.replace_sentinel_values(df)

        df = self.create_date_features(df)

        df = self.handle_missing_values(df)

        df = self.encode_categorical(
            df,
            training=True
        )

        df = self.remove_constant_columns(df)

        self.feature_columns = [
            c
            for c in df.columns
            if c != TARGET_COLUMN
        ]

        return df

    def transform(self, df):

        df = self.replace_sentinel_values(df)

        df = self.create_date_features(df)

        df = self.handle_missing_values(df)

        df = self.encode_categorical(
            df,
            training=False
        )

        missing_cols = set(
            self.feature_columns
        ) - set(df.columns)

        for col in missing_cols:
            df[col] = 0

        df = df.reindex(
            columns=self.feature_columns,
            fill_value=0
        )

        return df

    def save(self, path="model/preprocessor.pkl"):

        os.makedirs(
            os.path.dirname(path),
            exist_ok=True
        )

        joblib.dump(
            self,
            path
        )

    @staticmethod
    def load(path="model/preprocessor.pkl"):

        return joblib.load(path)


def split_features_target(df):

    X = df.drop(
        columns=[TARGET_COLUMN]
    )

    y = df[TARGET_COLUMN]

    return X, y