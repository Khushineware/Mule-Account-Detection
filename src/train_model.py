# src/train_model.py

import os
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

from sklearn.ensemble import (
    RandomForestClassifier,
    IsolationForest
)

from imblearn.over_sampling import SMOTE

from xgboost import XGBClassifier

from src.preprocess import (
    DataPreprocessor,
    split_features_target
)

from src.feature_engineering import (
    FeatureEngineer
)


DATA_PATH = "data/transactions.csv"

MODEL_DIR = "model"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "xgboost_model.pkl"
)

PREPROCESSOR_PATH = os.path.join(
    MODEL_DIR,
    "preprocessor.pkl"
)

FEATURE_PATH = os.path.join(
    MODEL_DIR,
    "feature_names.pkl"
)

ISO_PATH = os.path.join(
    MODEL_DIR,
    "isolation_forest.pkl"
)


os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


def evaluate_model(
    model,
    X_test,
    y_test,
    name
):

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions
    )

    recall = recall_score(
        y_test,
        predictions
    )

    f1 = f1_score(
        y_test,
        predictions
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC AUC  : {roc_auc:.4f}")

    print("\nClassification Report\n")

    print(
        classification_report(
            y_test,
            predictions
        )
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc
    }


def train_random_forest(
    X_train,
    y_train
):

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    rf.fit(
        X_train,
        y_train
    )

    return rf


def train_xgboost(
    X_train,
    y_train
):

    model = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="auc",
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    return model


def train_isolation_forest(
    X_train
):

    iso = IsolationForest(
        contamination=0.01,
        random_state=42,
        n_estimators=300
    )

    iso.fit(X_train)

    return iso


def perform_cross_validation(
    model,
    X,
    y
):

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    scores = cross_val_score(
        model,
        X,
        y,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1
    )

    print(
        "\nCross Validation ROC-AUC:"
    )

    print(scores)

    print(
        f"Mean ROC-AUC: {scores.mean():.4f}"
    )


def save_feature_importance(
    model,
    feature_names
):

    importance = pd.DataFrame(
        {
            "feature":
            feature_names,
            "importance":
            model.feature_importances_
        }
    )

    importance = (
        importance
        .sort_values(
            by="importance",
            ascending=False
        )
    )

    importance.head(
        50
    ).to_csv(
        "reports/top_features.csv",
        index=False
    )

    print(
        "\nTop feature file saved."
    )


def main():

    print(
        "\nLoading Dataset..."
    )

    raw_df = pd.read_csv(
        DATA_PATH
    )

    print(
        f"Shape: {raw_df.shape}"
    )

    print(
        "\nPreprocessing..."
    )

    preprocessor = (
        DataPreprocessor()
    )

    df = (
        preprocessor
        .fit_transform(raw_df)
    )

    print(
        "\nFeature Engineering..."
    )

    engineer = (
        FeatureEngineer()
    )

    df = engineer.transform(df)

    X, y = split_features_target(df)

    print(
        "\nTarget Distribution:"
    )

    print(
        y.value_counts()
    )

    print(
        "\nTrain Test Split..."
    )

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.2,
            stratify=y,
            random_state=42
        )
    )

    print(
        "\nApplying SMOTE..."
    )

    smote = SMOTE(
        random_state=42
    )

    X_train_smote, y_train_smote = (
        smote.fit_resample(
            X_train,
            y_train
        )
    )

    print(
        "\nAfter SMOTE:"
    )

    print(
        pd.Series(
            y_train_smote
        ).value_counts()
    )

    print(
        "\nTraining Random Forest..."
    )

    rf_model = train_random_forest(
        X_train_smote,
        y_train_smote
    )

    rf_metrics = evaluate_model(
        rf_model,
        X_test,
        y_test,
        "Random Forest"
    )

    print(
        "\nTraining XGBoost..."
    )

    xgb_model = train_xgboost(
        X_train_smote,
        y_train_smote
    )

    xgb_metrics = evaluate_model(
        xgb_model,
        X_test,
        y_test,
        "XGBoost"
    )

    # perform_cross_validation(
    #     xgb_model,
    #     X_train_smote,
    #     y_train_smote
    # )

    print(
        "\nTraining Isolation Forest..."
    )

    iso_model = (
        train_isolation_forest(
            X_train
        )
    )

    print(
        "\nSaving Models..."
    )

    joblib.dump(
        xgb_model,
        MODEL_PATH
    )

    joblib.dump(
        iso_model,
        ISO_PATH
    )

    preprocessor.save(
        PREPROCESSOR_PATH
    )

    joblib.dump(
        list(X.columns),
        FEATURE_PATH
    )
    
    joblib.dump(
    list(X.columns),
    "model/shap_features.pkl"
    )

    os.makedirs(
        "reports",
        exist_ok=True
    )

    save_feature_importance(
        xgb_model,
        list(X.columns)
    )

    metrics_file = open(
        "reports/metrics.txt",
        "w"
    )

    metrics_file.write(
        "Random Forest\n"
    )

    metrics_file.write(
        str(rf_metrics)
    )

    metrics_file.write(
        "\n\nXGBoost\n"
    )

    metrics_file.write(
        str(xgb_metrics)
    )

    metrics_file.close()

    print(
        "\nTraining Complete!"
    )

    print(
        f"\nModel saved at: {MODEL_PATH}"
    )


if __name__ == "__main__":
    main()