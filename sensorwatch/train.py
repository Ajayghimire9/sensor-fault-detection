"""Cost-aware sensor classification with a separate validation threshold."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline


def choose_threshold(y, probability, missed_fault_cost=10.0, false_alarm_cost=1.0):
    if missed_fault_cost <= 0 or false_alarm_cost <= 0:
        raise ValueError("Costs must be positive")
    best = None
    for threshold in np.linspace(0, 1, 101):
        pred = probability >= threshold
        cost = float(
            np.sum((y == 1) & ~pred) * missed_fault_cost
            + np.sum((y == 0) & pred) * false_alarm_cost
        )
        if best is None or cost < best[0]:
            best = cost, float(threshold)
    return best[1]


def train(frame, output):
    if "fault" not in frame or frame.empty or frame.columns.duplicated().any():
        raise ValueError("Expected unique sensor columns and binary fault label")
    y = frame.fault
    x = frame.drop(columns="fault").astype(float).replace([np.inf, -np.inf], np.nan)
    if (
        not y.isin([0, 1]).all()
        or y.nunique() != 2
        or x.shape[1] == 0
        or x.isna().all().any()
    ):
        raise ValueError("Invalid labels or entirely missing sensor")
    x_dev, x_test, y_dev, y_test = train_test_split(
        x, y, stratify=y, test_size=0.2, random_state=42
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_dev, y_dev, stratify=y_dev, test_size=0.25, random_state=42
    )
    model = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        RandomForestClassifier(
            n_estimators=100, class_weight="balanced", random_state=42, n_jobs=1
        ),
    )
    model.fit(x_train, y_train)
    threshold = choose_threshold(y_val.to_numpy(), model.predict_proba(x_val)[:, 1])
    probability = model.predict_proba(x_test)[:, 1]
    report = {
        "roc_auc": float(roc_auc_score(y_test, probability)),
        "pr_auc": float(average_precision_score(y_test, probability)),
        "threshold": threshold,
        "confusion_matrix": confusion_matrix(y_test, probability >= threshold).tolist(),
        "features": x.columns.tolist(),
    }
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "threshold": threshold, "features": x.columns.tolist()},
        out / "model.joblib",
    )
    (out / "metrics.json").write_text(json.dumps(report, indent=2))
    return report


def demo():
    x, y = make_classification(
        n_samples=500, n_features=8, weights=[0.9, 0.1], random_state=42
    )
    frame = pd.DataFrame(x, columns=[f"sensor_{i}" for i in range(8)])
    frame["fault"] = y
    return frame


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data",
        help="CSV with numeric sensor columns and fault=0/1; omit for synthetic demo",
    )
    p.add_argument("--output", default="artifacts")
    a = p.parse_args()
    result = train(pd.read_csv(a.data) if a.data else demo(), a.output)
    result["data_source"] = a.data or "synthetic smoke-test data"
    print(json.dumps(result, indent=2))
