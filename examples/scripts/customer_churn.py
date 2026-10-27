"""Predict customer churn from a CSV with standard CPU models.

Install: python -m pip install 'mljar-supervised==1.4.0b1'
Run: python customer_churn.py --output AutoML_CustomerChurn

Follow the numbered steps in run(). This file runs independently of notebooks
and docs tools. Use a new output directory for each run; see --help for options.
Data: Iranian Churn (2020), UCI, CC BY 4.0, https://doi.org/10.24432/C5JW3Z
Tutorial: https://supervised.mljar.com/tutorials/customer-churn/
"""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from urllib.request import urlopen

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, average_precision_score,
    brier_score_loss, confusion_matrix, log_loss, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits
from supervised import AutoML

DATA_URL = "https://archive.ics.uci.edu/static/public/563/data.csv"
DATA_SHA256 = "90d5fb6bd1630cd4de4b4d28fcf8b4cb92a8f6ab7484605b0799d47386f7dbe1"
FEATURES = [
    "Call Failure", "Complains", "Subscription Length", "Charge Amount",
    "Seconds of Use", "Frequency of use", "Frequency of SMS",
    "Distinct Called Numbers", "Tariff Plan", "Age",
]
TARGET = "Churn"
SEED = 42
THRESHOLD = 0.5  # Chosen before evaluation; not tuned on the test set.


def load_data(path):
    """Download the reviewed snapshot once; retain the original CSV unchanged."""
    if not path.exists():
        content = urlopen(DATA_URL, timeout=60).read()
        if hashlib.sha256(content).hexdigest() != DATA_SHA256:
            raise ValueError("UCI snapshot changed; review the dataset before continuing.")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    if hashlib.sha256(path.read_bytes()).hexdigest() != DATA_SHA256:
        raise ValueError("Dataset checksum differs from the reviewed snapshot.")
    data = pd.read_csv(path)
    data.columns = [" ".join(name.split()) for name in data.columns]
    expected = set(FEATURES + [TARGET, "Status", "Customer Value", "Age Group"])
    if data.shape != (3150, 14) or set(data.columns) != expected:
        raise ValueError("Unexpected churn dataset schema.")
    if not np.isfinite(data.to_numpy(dtype=float)).all():
        raise ValueError("The reviewed snapshot should contain only finite numbers.")
    if set(data[TARGET]) != {0, 1}:
        raise ValueError("Expected 0 (no churn) and 1 (churn).")
    return data


def evaluate(y, probabilities):
    """Use the same metrics and fixed threshold for baseline and AutoML."""
    labels = (probabilities >= THRESHOLD).astype(int)
    return {
        "roc_auc": roc_auc_score(y, probabilities),
        "average_precision": average_precision_score(y, probabilities),
        "log_loss": log_loss(y, probabilities, labels=[0, 1]),
        "brier_score": brier_score_loss(y, probabilities),
        "accuracy": accuracy_score(y, labels),
        "precision": precision_score(y, labels, zero_division=0),
        "recall": recall_score(y, labels, zero_division=0),
        "confusion_matrix": confusion_matrix(y, labels, labels=[0, 1]).tolist(),
    }


def run(output, data_path, budget):
    # Step 1. Inspect the customers and choose the target.
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=False)
    data = load_data(data_path)
    print(f"Customers: {len(data)}; churned: {data[TARGET].sum()} ({data[TARGET].mean():.1%}).", flush=True)
    print(data.head().to_string(index=False), flush=True)
    # Omit Status (proximity to churn), undocumented Customer Value, and redundant Age Group.
    X, y = data[FEATURES].copy(), data[TARGET]
    # Tariff codes name plans, not a numeric scale. AutoML handles their encoding.
    X["Tariff Plan"] = X["Tariff Plan"].map({1: "Pay as you go", 2: "Contractual"})

    # Step 2. Reserve test customers and keep identical input profiles together.
    # There are no customer IDs or dates. These groups are only duplicate-profile proxies.
    groups = X.groupby(FEATURES, sort=True, dropna=False).ngroup()
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    train, test = next(outer.split(X, y, groups))
    X_train, y_train = X.iloc[train].reset_index(drop=True), y.iloc[train].reset_index(drop=True)
    X_test, y_test = X.iloc[test], y.iloc[test]
    train_groups = groups.iloc[train].reset_index(drop=True)
    cv = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED).split(
        X_train, y_train, train_groups))
    assert set(groups.iloc[train]).isdisjoint(groups.iloc[test])
    for fit_rows, valid_rows in cv:
        assert set(train_groups.iloc[fit_rows]).isdisjoint(train_groups.iloc[valid_rows])
    split_record = {
        "seed": SEED, "train_rows": train.tolist(), "test_rows": test.tolist(),
        "profile_groups": groups.tolist(),
        "cv_indices_relative_to_train": [
            {"train": a.tolist(), "validation": b.tolist()} for a, b in cv
        ],
    }
    (output / "splits.json").write_text(json.dumps(split_record, indent=2) + "\n")
    print(f"Train: {len(train)} rows; test: {len(test)} rows ({y_test.mean():.1%} churn).", flush=True)

    # Step 3. Train a constant-probability baseline and compare AutoML models.
    baseline = DummyClassifier(strategy="prior").fit(X_train, y_train)
    automl = AutoML(
        results_path=str(output / "automl"), mode="Explain",
        algorithms=["Baseline", "Linear", "Decision Tree", "LightGBM"],
        ml_task="binary_classification", eval_metric="logloss", total_time_limit=budget,
        validation_strategy={"validation_type": "custom"},
        feature_discovery=False, golden_features=False, kmeans_features=False,
        features_selection=False, stack_models=False, train_ensemble=True,
        explain_level=1, n_jobs=2, random_state=SEED,
    )
    fit_started = time.perf_counter()
    automl.fit(X_train, y_train, cv=cv)
    fit_seconds = time.perf_counter() - fit_started

    # Step 4. Evaluate probabilities and classification errors on untouched rows.
    # For our numeric target 0/1, the second probability column is class 1 (churn).
    probabilities = automl.predict_proba(X_test)[:, 1]
    baseline_probabilities = baseline.predict_proba(X_test)[:, 1]
    scores = {"baseline": evaluate(y_test, baseline_probabilities),
              "automl": evaluate(y_test, probabilities)}
    print(pd.DataFrame(scores).drop(index="confusion_matrix").to_string(), flush=True)
    X_test.assign(source_row=test, actual_churn=y_test,
                  churn_probability=probabilities,
                  predicted_churn=(probabilities >= THRESHOLD).astype(int),
                  baseline_probability=baseline_probabilities).to_csv(output / "test_predictions.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), layout="constrained")
    for ax, (name, result) in zip(axes, scores.items()):
        ConfusionMatrixDisplay(np.array(result["confusion_matrix"]),
                               display_labels=["Stayed", "Churned"]).plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"{name.title()} · threshold {THRESHOLD}")
    fig.savefig(output / "confusion-matrix.png", dpi=160)
    plt.close(fig)

    # Step 5. Predict example inputs and verify reload in a fresh interpreter.
    # These are held-out examples with the label removed, not newly collected customers.
    new_rows = X_test.head(3)
    new_rows.to_csv(output / "new_customers.csv", index=False)
    print("Example churn probabilities:", automl.predict_proba(new_rows)[:, 1], flush=True)
    X_test.to_csv(output / "reload_inputs.csv", index=False)
    subprocess.run([
        sys.executable, "-c", textwrap.dedent("""
        import sys
        import numpy as np
        import pandas as pd
        from supervised import AutoML
        model = AutoML(results_path=sys.argv[1])
        inputs = pd.read_csv(sys.argv[2])
        np.save(sys.argv[3], model.predict_proba(inputs)[:, 1])
        """),
        str(output / "automl"), str(output / "reload_inputs.csv"),
        str(output / "reloaded_probabilities.npy"),
    ], check=True)
    reloaded = np.load(output / "reloaded_probabilities.npy")
    np.testing.assert_allclose(probabilities, reloaded, rtol=1e-7, atol=1e-7)
    print("Fresh-process reload agrees with the original probabilities.", flush=True)

    # Step 6. Create the HTML report and record this run.
    report = automl.report()
    metrics = {
        "dataset_url": DATA_URL, "dataset_sha256": DATA_SHA256,
        "features": FEATURES, "seed": SEED, "threshold": THRESHOLD,
        "train_rows": len(train), "test_rows": len(test),
        "train_churned": int(y_train.sum()), "test_churned": int(y_test.sum()),
        "profile_groups": int(groups.nunique()), "scores": scores,
        "training_budget_seconds": budget, "fit_wall_seconds": fit_seconds,
        "workflow_seconds_excluding_imports": time.perf_counter() - started,
        "reload_max_absolute_difference": float(np.max(np.abs(probabilities - reloaded))),
        "python": platform.python_version(), "platform": platform.platform(),
        "package_versions": {name: importlib.metadata.version(name) for name in (
            "mljar-supervised", "numpy", "pandas", "scikit-learn", "lightgbm", "features-goldmine", "joblib")},
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"Open the report: {(output / 'automl' / 'README.html').as_uri()}", flush=True)
    print("Keep the complete automl directory for future predictions.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("AutoML_CustomerChurn"))
    parser.add_argument("--data", type=Path, default=Path("AutoML_ChurnData/customer-churn.csv"))
    parser.add_argument("--budget", type=int, default=120, help="Training budget in seconds")
    args = parser.parse_args()
    if args.budget <= 0:
        parser.error("--budget must be positive")
    with threadpool_limits(limits=2):
        run(args.output.resolve(), args.data.resolve(), args.budget)
