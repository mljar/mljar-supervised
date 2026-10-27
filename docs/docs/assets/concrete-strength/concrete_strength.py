"""Predict a number from a CSV: concrete strength from ingredients and age.

Install: python -m pip install 'mljar-supervised==1.4.0b1'
Run from the folder containing this file:
    python concrete_strength.py --output AutoML_ConcreteStrength

Follow the numbered steps in run() below. This standalone script downloads the
data, trains CPU models, evaluates unseen mixtures, verifies reload predictions,
and saves an HTML report. It needs no notebook or documentation-building tools.
Use a new output directory for each run. See --help for data and budget options.

Data: Yeh, I. (1998), Concrete Compressive Strength, UCI (CC BY 4.0).
Source and attribution: https://doi.org/10.24432/C5PK67
Tutorial: https://supervised.mljar.com/tutorials/concrete-strength/
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
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from threadpoolctl import threadpool_limits

from supervised import AutoML


DATA_URL = "https://archive.ics.uci.edu/static/public/165/data.csv"
DATA_SHA256 = "8d4b15b6fc68cd932d745cbd663d5ceae66dd54422e99c1e4865f2936ab7e2af"
INGREDIENTS = [
    "Cement", "Blast Furnace Slag", "Fly Ash", "Water", "Superplasticizer",
    "Coarse Aggregate", "Fine Aggregate",
]
FEATURES = INGREDIENTS + ["Age"]
TARGET = "Concrete compressive strength"
SEED = 42


def load_data(path):
    """Download once, then check the exact UCI snapshot on every run."""
    if not path.exists():
        with urlopen(DATA_URL, timeout=60) as response:
            content = response.read()
        if hashlib.sha256(content).hexdigest() != DATA_SHA256:
            raise ValueError("UCI data changed; review the new snapshot before using it.")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    if hashlib.sha256(path.read_bytes()).hexdigest() != DATA_SHA256:
        raise ValueError("Dataset checksum differs from the reviewed UCI snapshot.")
    data = pd.read_csv(path)
    if list(data.columns) != FEATURES + [TARGET] or data.shape != (1030, 9):
        raise ValueError("Unexpected concrete dataset schema.")
    if not np.isfinite(data.to_numpy(dtype=float)).all():
        raise ValueError("The reviewed dataset should contain only finite numbers.")
    return data


def run(output, data_path, budget):
    # Step 1. Load the measurements and choose inputs and target.
    # Seven ingredient quantities (kg/m³) and age (days) predict strength (MPa).
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=False)
    data = load_data(data_path)
    X, y = data[FEATURES], data[TARGET]

    # Keep the same mixture at different ages on one side of every split (step 2).
    # These are ingredient-based proxy groups, not verified experiment IDs.
    groups = data.groupby(INGREDIENTS, sort=True).ngroup()
    train, test = next(GroupShuffleSplit(
        n_splits=1, test_size=0.2, random_state=SEED
    ).split(X, y, groups))
    X_train, y_train = X.iloc[train].reset_index(drop=True), y.iloc[train].reset_index(drop=True)
    X_test, y_test = X.iloc[test], y.iloc[test]
    train_groups = groups.iloc[train].reset_index(drop=True)
    cv = list(GroupKFold(n_splits=3).split(X_train, y_train, train_groups))
    assert set(groups.iloc[train]).isdisjoint(groups.iloc[test])
    for fit_rows, valid_rows in cv:
        assert set(train_groups.iloc[fit_rows]).isdisjoint(train_groups.iloc[valid_rows])

    split_record = {
        "seed": SEED, "train_rows": train.tolist(), "test_rows": test.tolist(),
        "mixture_groups": groups.tolist(),
        "cv_indices_relative_to_train": [
            {"train": a.tolist(), "validation": b.tolist()} for a, b in cv
        ],
    }
    (output / "splits.json").write_text(json.dumps(split_record, indent=2) + "\n")
    print(f"Concrete: {len(data)} measurements, {groups.nunique()} mixture groups.", flush=True)
    print(f"Train: {len(train)} rows; held-out test: {len(test)} rows. Target: MPa.", flush=True)
    print(data.head().to_string(index=False), flush=True)

    # A training median is our baseline (step 3): the test target is unused.
    # AutoML compares models using only the three training-validation folds.
    baseline = DummyRegressor(strategy="median").fit(X_train, y_train)
    automl = AutoML(
        results_path=str(output / "automl"), mode="Explain",
        algorithms=["Baseline", "Linear", "Decision Tree", "LightGBM"],
        ml_task="regression", eval_metric="mae", total_time_limit=budget,
        validation_strategy={"validation_type": "custom"},
        feature_discovery=False, golden_features=False, kmeans_features=False,
        features_selection=False, stack_models=False, train_ensemble=True,
        explain_level=1, n_jobs=2, random_state=SEED,
    )
    fit_started = time.perf_counter()
    automl.fit(X_train, y_train, cv=cv)
    fit_seconds = time.perf_counter() - fit_started
    # Step 4. Evaluate on rows excluded from training and model selection.
    predictions = automl.predict(X_test)
    baseline_predictions = baseline.predict(X_test)
    print(f"Baseline test MAE: {mean_absolute_error(y_test, baseline_predictions):.2f} MPa", flush=True)
    print(f"AutoML test MAE: {mean_absolute_error(y_test, predictions):.2f} MPa", flush=True)
    X_test.assign(
        source_row=test, measured_strength_mpa=y_test,
        predicted_strength_mpa=predictions, baseline_strength_mpa=baseline_predictions,
    ).to_csv(output / "test_predictions.csv", index=False)

    # Demonstrate new, unlabeled inputs using held-out examples (not invented designs).
    new_rows = X_test.head(3)
    new_rows.to_csv(output / "new_rows.csv", index=False)
    new_predictions = automl.predict(new_rows)
    print("Example input predictions (MPa):", new_predictions, flush=True)

    # A new interpreter loads only the saved model directory and input CSV (step 5).
    # This checks that future predictions work without our training-session state.
    X_test.to_csv(output / "reload_inputs.csv", index=False)
    subprocess.run([
        sys.executable, "-c",
        textwrap.dedent("""
        import sys
        import numpy as np
        import pandas as pd
        from supervised import AutoML

        model = AutoML(results_path=sys.argv[1])
        inputs = pd.read_csv(sys.argv[2])
        np.save(sys.argv[3], model.predict(inputs))
        """),
        str(output / "automl"), str(output / "reload_inputs.csv"),
        str(output / "reloaded_predictions.npy"),
    ], check=True)
    reloaded = np.load(output / "reloaded_predictions.npy")
    np.testing.assert_allclose(predictions, reloaded, rtol=1e-7, atol=1e-7)
    print("Fresh-process reload agrees with the original predictions.", flush=True)

    # Step 6. Plot errors and save the evaluation and environment record.
    fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
    ax.scatter(y_test, predictions, s=20, alpha=0.65)
    low, high = min(y_test.min(), predictions.min()), max(y_test.max(), predictions.max())
    ax.plot([low, high], [low, high], color="black", linestyle="--", label="Perfect prediction")
    ax.set(xlabel="Measured strength (MPa)", ylabel="Predicted strength (MPa)",
           title="Concrete strength: held-out mixture groups")
    ax.legend()
    fig.savefig(output / "predicted-vs-measured.png", dpi=160)
    plt.close(fig)

    metrics = {
        "dataset_url": DATA_URL, "dataset_sha256": DATA_SHA256,
        "train_rows": len(train), "test_rows": len(test),
        "train_groups": int(groups.iloc[train].nunique()),
        "test_groups": int(groups.iloc[test].nunique()),
        "baseline_strategy": "training median",
        "baseline_mae_mpa": mean_absolute_error(y_test, baseline_predictions),
        "automl_mae_mpa": mean_absolute_error(y_test, predictions),
        "training_budget_seconds": budget, "fit_wall_seconds": fit_seconds,
        "workflow_seconds_excluding_imports": time.perf_counter() - started,
        "reload_max_absolute_difference": float(np.max(np.abs(predictions - reloaded))),
        "python": platform.python_version(), "platform": platform.platform(),
        "package_versions": {name: importlib.metadata.version(name) for name in (
            "mljar-supervised", "numpy", "pandas", "scikit-learn", "lightgbm",
            "features-goldmine", "joblib",
        )},
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)

    # Step 7. Create the HTML report and print its browser location.
    report = automl.report()
    print(f"Open the run report: {(output / 'automl' / 'README.html').as_uri()}", flush=True)
    print("Keep the complete automl directory for future predictions.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("AutoML_ConcreteStrength"))
    parser.add_argument("--data", type=Path, default=Path("AutoML_ConcreteData/concrete.csv"))
    parser.add_argument("--budget", type=int, default=120, help="Training budget in seconds")
    args = parser.parse_args()
    if args.budget <= 0:
        parser.error("--budget must be positive")
    with threadpool_limits(limits=2):
        run(args.output.resolve(), args.data.resolve(), args.budget)
