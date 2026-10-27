"""Keep repeated measurements from a machine together.

Synthetic regression recipe, not a measured industry dataset or forecast benchmark.
Install: python -m pip install 'mljar-supervised==1.4.0b1'
Run: python custom_validation_grouped.py --output AutoML_GroupedValidation
Uses CPU models, 60-second training budget, two threads, and no downloads.
Use a fresh output directory. Follow the numbered steps in run().
"""

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
import textwrap
from pathlib import Path

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

SEED = 42

def run(output):
    output.mkdir(parents=True, exist_ok=False)
    # Step 1. Generate repeated synthetic measurements from 60 machines.
    rng = np.random.default_rng(SEED)
    machine = np.repeat(np.arange(60), 8)
    signal = rng.normal(size=len(machine))
    machine_effect = rng.normal(scale=2, size=60)
    data = pd.DataFrame({"machine_id": machine, "sensor": signal,
                         "load": rng.uniform(0, 1, len(machine))})
    data["response"] = 4 * data.sensor + 3 * data.load + machine_effect[machine] + rng.normal(scale=.3, size=len(machine))
    data.to_csv(output / "synthetic_data.csv", index=False)
    X, y, groups = data[["sensor", "load"]], data.response, data.machine_id
    print("Synthetic example: 60 machines, 8 measurements each. IDs are not model inputs.", flush=True)

    # Step 2. Hold out entire machines before creating model-selection folds.
    # docs:split:start
    train, test = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=SEED).split(X, y, groups))
    X_train = X.iloc[train].reset_index(drop=True)
    y_train = y.iloc[train].reset_index(drop=True)
    train_groups = groups.iloc[train].reset_index(drop=True)
    X_test, y_test = X.iloc[test], y.iloc[test]
    cv = list(GroupKFold(n_splits=3).split(X_train, y_train, train_groups))
    assert set(groups.iloc[train]).isdisjoint(groups.iloc[test])
    for fit_rows, valid_rows in cv:
        assert set(fit_rows).isdisjoint(valid_rows)
        assert set(train_groups.iloc[fit_rows]).isdisjoint(train_groups.iloc[valid_rows])
    # docs:split:end

    # Step 3. Save exact indices and a diagram so the split can be inspected.
    record = {"outer_train": train.tolist(), "outer_test": test.tolist(),
              "groups": groups.tolist(), "cv_indices_relative_to_train": [
                  {"train": a.tolist(), "validation": b.tolist()} for a, b in cv]}
    (output / "splits.json").write_text(json.dumps(record, indent=2) + "\n")
    from matplotlib.colors import ListedColormap
    plan = np.full((4, 60), -1)
    plan[0, groups.iloc[train].unique()] = 0
    plan[0, groups.iloc[test].unique()] = 2
    for i, (fit_rows, valid_rows) in enumerate(cv, start=1):
        plan[i, train_groups.iloc[fit_rows].unique()] = 0
        plan[i, train_groups.iloc[valid_rows].unique()] = 1
    fig, ax = plt.subplots(figsize=(10, 3.2), layout="constrained")
    ax.imshow(plan, aspect="auto", interpolation="nearest", vmin=-1, vmax=2,
              cmap=ListedColormap(["#f1f5f9", "#0099cc", "#97d088", "#ef6f6c"]))
    ax.set(yticks=range(4), yticklabels=["Final split", "Fold 1", "Fold 2", "Fold 3"],
           xlabel="Synthetic machine ID (all 8 rows stay together)",
           title="Blue: training · Green: validation · Coral: final test · Gray: unused")
    fig.savefig(output / "splits.png", dpi=160); plt.close(fig)
    print(f"Development: {len(X_train)} rows / {train_groups.nunique()} machines; test: {len(X_test)} rows / {groups.iloc[test].nunique()} machines.", flush=True)

    # Step 4. Train using only the supplied development-data folds.
    automl = AutoML(
        results_path=str(output / "automl"), mode="Explain",
        algorithms=["Baseline", "Linear", "Decision Tree"],
        ml_task="regression", eval_metric="mae", total_time_limit=60,
        validation_strategy={"validation_type": "custom"},
        feature_discovery=False, golden_features=False, kmeans_features=False,
        features_selection=False, train_ensemble=False, stack_models=False,
        explain_level=1, n_jobs=2, random_state=SEED,
    )
    automl.fit(X_train, y_train, cv=cv)

    # Step 5. Evaluate once on the reserved test rows.
    baseline = DummyRegressor(strategy="median").fit(X_train, y_train)
    predictions = automl.predict(X_test)
    baseline_predictions = baseline.predict(X_test)
    X_test.assign(actual=y_test, predicted=predictions,
                  baseline=baseline_predictions).to_csv(output / "test_predictions.csv", index=False)
    X_test.to_csv(output / "test_inputs.csv", index=False)
    print(f"Baseline test MAE: {mean_absolute_error(y_test, baseline_predictions):.4f}", flush=True)
    print(f"AutoML test MAE: {mean_absolute_error(y_test, predictions):.4f}", flush=True)

    # Step 6. Verify the saved model in a new Python process and open its report.
    subprocess.run([
        sys.executable, "-c", textwrap.dedent("""
        import sys
        import numpy as np
        import pandas as pd
        from supervised import AutoML
        model = AutoML(results_path=sys.argv[1])
        np.save(sys.argv[3], model.predict(pd.read_csv(sys.argv[2])))
        """), str(output / "automl"), str(output / "test_inputs.csv"),
        str(output / "reloaded_predictions.npy"),
    ], check=True)
    reloaded = np.load(output / "reloaded_predictions.npy")
    np.testing.assert_allclose(predictions, reloaded, rtol=1e-7, atol=1e-7)
    automl.report()
    metrics = {
        "synthetic": True, "seed": SEED, "train_rows": len(X_train),
        "test_rows": len(X_test), "folds": len(cv),
        "baseline_mae": mean_absolute_error(y_test, baseline_predictions),
        "automl_mae": mean_absolute_error(y_test, predictions),
        "reload_max_absolute_difference": float(np.max(np.abs(predictions - reloaded))),
        "python": platform.python_version(),
        "package_versions": {name: importlib.metadata.version(name) for name in
                             ("mljar-supervised", "scikit-learn", "numpy", "pandas")},
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print("Fresh-process reload agrees with the original predictions.", flush=True)
    print(f"Split diagram: {output / 'splits.png'}", flush=True)
    print(f"Open report: {(output / 'automl' / 'README.html').as_uri()}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("AutoML_GroupedValidation"))
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        run(args.output.resolve())
