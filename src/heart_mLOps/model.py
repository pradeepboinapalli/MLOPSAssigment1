"""Leakage-safe preprocessing and model candidates."""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

from .schema import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("numeric", numeric, NUMERIC_FEATURES),
        ("categorical", categorical, CATEGORICAL_FEATURES),
    ])


def build_pipeline(estimator) -> Pipeline:
    return Pipeline([("preprocess", build_preprocessor()), ("classifier", estimator)])


def candidate_models(random_state: int = 42) -> dict:
    return {
        "logistic_regression": build_pipeline(LogisticRegression(solver="liblinear", max_iter=2000, class_weight="balanced", random_state=random_state)),
        "random_forest": build_pipeline(RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced", random_state=random_state, n_jobs=-1)),
    }
