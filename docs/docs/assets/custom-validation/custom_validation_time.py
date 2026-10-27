"""Train on earlier daily rows and validate on later rows.

Synthetic regression recipe, not a measured industry dataset or forecast benchmark.
Install: python -m pip install 'mljar-supervised==1.4.0b1'
Run: python custom_validation_time.py --output AutoML_TimeValidation
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
from sklearn.model_selection import TimeSeriesSplit
from threadpoolctl import threadpool_limits
from supervised import AutoML

SEED = 42

GAP = 7  # Seven daily rows, not an automatic choice for other datasets.


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    # Step 1. Generate a daily synthetic process, with inputs known in advance.
    rng = np.random.default_rng(SEED)
    day = np.arange(360)
    dates = pd.date_range("2025-01-01", periods=len(day), freq="D")
    planned_load = rng.uniform(0, 1, len(day))
    weekend = (dates.dayofweek >= 5).astype(int)
    data = pd.DataFrame({"date": dates, "day": day, "weekend": weekend,
                         "planned_load": planned_load})
    data["response"] = 20 + .03 * day + 5 * weekend + 3 * planned_load + rng.normal(scale=.5, size=len(day))
    data = data.sort_values("date").reset_index(drop=True)
    data.to_csv(output / "synthetic_data.csv", index=False)
    X, y = data[["day", "weekend", "planned_load"]], data.response
    print("Synthetic daily example. Calendar and planned load are known before prediction.", flush=True)

    # Step 2. Reserve the last 60 days, leaving a seven-day gap before them.
    # docs:split:start
    test_start = len(data) - 60
    train = np.arange(test_start - GAP)
    test = np.arange(test_start, len(data))
    X_train, y_train = X.iloc[train].reset_index(drop=True), y.iloc[train].reset_index(drop=True)
    X_test, y_test = X.iloc[test], y.iloc[test]
    dates_train = data.date.iloc[train].reset_index(drop=True)
    cv = list(TimeSeriesSplit(n_splits=3, test_size=40, gap=GAP).split(X_train))
    assert train.max() + GAP < test.min()
    assert data.date.iloc[train].max() < data.date.iloc[test].min()
    for fit_rows, valid_rows in cv:
        assert set(fit_rows).isdisjoint(valid_rows)
        assert fit_rows.max() + GAP < valid_rows.min()
        assert dates_train.iloc[fit_rows].max() < dates_train.iloc[valid_rows].min()
    # docs:split:end

    # Step 3. Record windows and draw the actual split boundaries.
    record = {"outer_train": train.tolist(), "outer_test": test.tolist(),
              "gap_rows": GAP, "dates": data.date.dt.strftime("%Y-%m-%d").tolist(),
              "cv_indices_relative_to_train": [
                  {"train": a.tolist(), "validation": b.tolist()} for a, b in cv]}
    (output / "splits.json").write_text(json.dumps(record, indent=2) + "\n")
    from matplotlib.patches import Patch
    fig, ax = plt.subplots(figsize=(10, 3.2), layout="constrained")
    ax.broken_barh([(0, len(train))], (-.3, .6), facecolors="#0099cc")
    ax.broken_barh([(len(train), GAP)], (-.3, .6), facecolors="#bfd7ea")
    ax.broken_barh([(test_start, len(test))], (-.3, .6), facecolors="#ef6f6c")
    for i, (fit_rows, valid_rows) in enumerate(cv, start=1):
        ax.broken_barh([(0, len(fit_rows))], (i-.3, .6), facecolors="#0099cc")
        ax.broken_barh([(len(fit_rows), GAP)], (i-.3, .6), facecolors="#bfd7ea")
        ax.broken_barh([(valid_rows.min(), len(valid_rows))], (i-.3, .6), facecolors="#97d088")
        print(f"Fold {i}: train days {fit_rows.min()}–{fit_rows.max()}, gap {GAP}, validate days {valid_rows.min()}–{valid_rows.max()}.", flush=True)
    ax.set(yticks=range(4), yticklabels=["Final split", "Fold 1", "Fold 2", "Fold 3"],
           xlabel="Day position (earlier → later)", xlim=(0, len(data)))
    ax.invert_yaxis()
    ax.legend(handles=[Patch(color=c, label=l) for c, l in [
        ("#0099cc", "Training"), ("#bfd7ea", "Gap"), ("#97d088", "Validation"),
        ("#ef6f6c", "Final test")]], loc="upper center", bbox_to_anchor=(.5, 1.24), ncol=4)
    fig.savefig(output / "splits.png", dpi=160); plt.close(fig)
    print(f"Development: {len(train)} days; gap: {GAP}; final test: {len(test)} days.", flush=True)

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
    parser.add_argument("--output", type=Path, default=Path("AutoML_TimeValidation"))
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        run(args.output.resolve())
