"""Train, compare and log model candidates with cross-validation and MLflow."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report,
                             precision_score, recall_score, roc_auc_score, RocCurveDisplay)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from .model import candidate_models
from .schema import FEATURES, TARGET


def train(data_path: Path, output_dir: Path, tracking_uri: str | None = None) -> dict:
    data = pd.read_csv(data_path)
    X, y = data[FEATURES], data[TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {"accuracy": "accuracy", "precision": "precision", "recall": "recall", "roc_auc": "roc_auc"}
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("uci-heart-disease-four-site")
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = {}
    fitted = {}
    for name, model in candidate_models().items():
        with mlflow.start_run(run_name=name):
            mlflow.log_params({"model": name, "cv_folds": 5, "test_size": 0.2, "seed": 42, "feature_count": len(FEATURES)})
            scores = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
            cv_metrics = {f"cv_{metric}": float(scores[f"test_{metric}"].mean()) for metric in scoring}
            model.fit(X_train, y_train)
            probabilities = model.predict_proba(X_test)[:, 1]
            predictions = (probabilities >= 0.5).astype(int)
            test_metrics = {
                "test_accuracy": accuracy_score(y_test, predictions),
                "test_precision": precision_score(y_test, predictions, zero_division=0),
                "test_recall": recall_score(y_test, predictions, zero_division=0),
                "test_roc_auc": roc_auc_score(y_test, probabilities),
            }
            metrics = {**cv_metrics, **{k: float(v) for k, v in test_metrics.items()}}
            mlflow.log_metrics(metrics)
            RocCurveDisplay.from_predictions(y_test, probabilities)
            plt.title(f"ROC curve - {name}")
            roc_path = output_dir / f"roc_{name}.png"
            plt.savefig(roc_path, bbox_inches="tight", dpi=140)
            plt.close()
            report_path = output_dir / f"report_{name}.json"
            report_path.write_text(json.dumps(classification_report(y_test, predictions, output_dict=True), indent=2), encoding="utf-8")
            mlflow.log_artifact(str(roc_path))
            mlflow.log_artifact(str(report_path))
            input_example = X_train.head(5)
            mlflow.sklearn.log_model(sk_model=model,
            artifact_path="model",
            input_example=input_example,)
            summaries[name] = metrics
            fitted[name] = model
    # Recall is selected as the primary selection metric because missed positive cases
    # are especially costly in screening; ROC-AUC breaks ties.
    selected_name = max(summaries, key=lambda n: (summaries[n]["cv_recall"], summaries[n]["cv_roc_auc"]))
    final_model = fitted[selected_name]
    model_path = output_dir / "heart_disease_model.joblib"
    joblib.dump(final_model, model_path)
    metadata = {"selected_model": selected_name, "selection_metric": "mean 5-fold CV recall, tie-break CV ROC-AUC", "metrics": summaries, "features": FEATURES}
    (output_dir / "training_summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed/heart_disease.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI"))
    args = parser.parse_args()
    print(json.dumps(train(args.data, args.output_dir, args.tracking_uri), indent=2))


if __name__ == "__main__":
    main()
