"""
Model definitions and persistence.

Each build_* function returns an UNFITTED Pipeline (preprocessor + classifier),
so it can be used both for cross-validation and for the final fit.
"""
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from xgboost import XGBClassifier   # pip install xgboost

RANDOM_STATE = 42


def build_logistic(preprocessor) -> Pipeline:
    """Logistic regression with balanced class weights."""
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=200,
            class_weight="balanced",
        )),
    ])


def build_random_forest(preprocessor) -> Pipeline:
    """Random Forest with balanced class weights."""
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            random_state=RANDOM_STATE,
            class_weight="balanced"
        )),
    ])

def build_svm(preprocessor) -> Pipeline:
    """Support Vector Machine with balanced class weights."""
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced"
        )),
    ])


def build_xgboost(preprocessor, scale_pos_weight: float) -> Pipeline:
    """XGBoost; scale_pos_weight plays the role of class_weight='balanced'."""
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
        )),
    ])


def get_candidates(preprocessor, y_train) -> dict:
    """All candidate models, keyed by display name."""
    # Class imbalance computed from the TRAIN labels only
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()

    return {
        "Logistic Regression": build_logistic(preprocessor),
        "Random Forest": build_random_forest(preprocessor),
        "SVM": build_svm(preprocessor),
        "XGBoost": build_xgboost(preprocessor, scale_pos_weight=neg / pos),
    }


def save_model(model, path) -> None:
    """Save a fitted pipeline with joblib, creating the folder if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved in {path}")