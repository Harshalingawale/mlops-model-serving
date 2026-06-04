"""Train a tabular classifier and persist the model + metrics.

Uses scikit-learn's built-in breast-cancer dataset so the project trains with
zero external downloads. Swap `load_data()` for your own CSV in production.

    python -m src.train
"""
from __future__ import annotations
import json
from pathlib import Path

import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)
MODEL_PATH = ARTIFACTS / "model.joblib"
METRICS_PATH = ARTIFACTS / "metrics.json"
FEATURES_PATH = ARTIFACTS / "features.json"


def load_data():
    ds = load_breast_cancer(as_frame=True)
    return ds.data, ds.target, list(ds.feature_names)


def main() -> dict:
    X, y, feature_names = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)),
    ])
    pipe.fit(X_train, y_train)

    proba = pipe.predict_proba(X_test)[:, 1]
    preds = pipe.predict(X_test)
    cv = cross_val_score(pipe, X, y, cv=5, scoring="roc_auc")

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, preds)), 4),
        "f1": round(float(f1_score(y_test, preds)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
        "cv_roc_auc_mean": round(float(cv.mean()), 4),
        "cv_roc_auc_std": round(float(cv.std()), 4),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

    joblib.dump(pipe, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    FEATURES_PATH.write_text(json.dumps(feature_names, indent=2))
    print("Saved model ->", MODEL_PATH)
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    main()
