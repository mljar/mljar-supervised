"""One published-version fit, with retained models and separately timed test evaluation."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score, root_mean_squared_error
from sklearn.preprocessing import LabelEncoder

from common import CANDIDATE_VERSIONS, NEW_ALGORITHMS, read_json, sha256, write_json


def probability_matrix(frame, n_classes):
    """Align numeric encoded labels, including lexically ordered classes 10+ ."""
    columns = [column for column in frame if str(column).startswith("prediction_")]
    codes = [int(str(column).removeprefix("prediction_")) for column in columns]
    if sorted(codes) != list(range(n_classes)):
        raise ValueError(f"Unexpected probability columns: {columns}")
    values = frame[columns].to_numpy(dtype=float)[:, np.argsort(codes)]
    if (not np.isfinite(values).all() or (values < 0).any() or (values > 1).any()
            or not np.allclose(values.sum(axis=1), 1, atol=1e-5)):
        raise ValueError("Invalid class probabilities")
    return values


def score_predictions(frame, y, problem_type, n_classes=None):
    if problem_type == "regression":
        values = frame["prediction"].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Non-finite regression predictions")
        score = root_mean_squared_error(y, values)
        return float(score), float(score)
    values = probability_matrix(frame, n_classes)
    if problem_type == "binary":
        score = float(roc_auc_score(y, values[:, 1]))
        return score, 1 - score
    score = float(log_loss(y, values, labels=np.arange(n_classes)))
    return score, score


def save_table(path, rows):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    pd.DataFrame(rows).to_csv(temporary, index=False)
    temporary.replace(path)


def model_inventory(automl, root, leaderboard):
    records = []
    lookup = {row["name"]: row for row in leaderboard.to_dict("records")}
    for model in automl._models:
        name = model.get_name()
        folder = root / name
        framework = read_json(folder / "framework.json") if (folder / "framework.json").exists() else {}
        ensemble = read_json(folder / "ensemble.json") if (folder / "ensemble.json").exists() else {}
        learner = framework.get("params", {}).get("learner", {})
        training_info = [item.get("training_info", {}) for item in framework.get("learners", [])]
        devices = sorted({item.get("device", item.get("resources", {}).get("device", "unknown"))
                          for item in training_info})
        records.append(dict(lookup.get(name, {}), name=name, algorithm=model.get_type(),
                            is_stacked=bool(model._is_stacked), model_directory=str(folder),
                            training_devices=",".join(devices) if any(training_info) else "see learner_parameters",
                            learner_parameters=json.dumps(learner, sort_keys=True),
                            training_info=json.dumps(training_info, sort_keys=True),
                            ensemble_members=json.dumps(ensemble.get("selected_models", [])),
                            test_status="pending"))
    return records


def coverage(root, inventory):
    progress = read_json(root / "progress.json") if (root / "progress.json").exists() else {}
    candidates = [item for group in progress.get("all_params", {}).values() for item in group]
    results = {}
    for algorithm in NEW_ALGORITHMS:
        selected = [p for p in candidates if p.get("learner", {}).get("model_type") == algorithm]
        completed = [r["name"] for r in inventory if r["algorithm"] == algorithm]
        results[algorithm] = dict(completed_models=completed,
                                  candidates=[dict(name=p.get("name"), status=p.get("status"),
                                                   final_loss=p.get("final_loss"), train_time=p.get("train_time"))
                                              for p in selected],
                                  outcome="completed" if completed else "no_completed_model",
                                  details="See automl/errors.md, progress.json and console.log for candidate failures/skips.")
    return results


def main(spec):
    import supervised
    from supervised import AutoML

    version = importlib.metadata.version("mljar-supervised")
    if version != spec["version"] or supervised.__version__ != version:
        raise RuntimeError(f"Wrong version: {version} / {supervised.__version__}")
    if not Path(supervised.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError(f"Workspace shadowed the installed package: {supervised.__file__}")
    data, output = Path(spec["data"]), Path(spec["attempt"])
    if sha256(data / "dataset.json") != spec["dataset_sha256"]:
        raise RuntimeError("Dataset manifest changed after preparation")
    info = read_json(data / "dataset.json")
    for name, digest in info["files_sha256"].items():
        if sha256(data / name) != digest:
            raise RuntimeError(f"Dataset file changed: {name}")
    write_json(output / "dataset.json", info)
    print(f"Dataset: {info['dataset']} | {info['domain']}\nSource: {info['source_url']}", flush=True)
    print(info["description"], flush=True)
    print(f"Official split {info['split']}: {info['train_rows']:,} training / {info['test_rows']:,} test rows\n"
          f"Features: {info['features']} ({info['numerical_features']} numerical, {info['categorical_features']} categorical)\n"
          f"Missing training cells: {info['training_missing_cells']:,}\n"
          f"Target: {info['target']} | {info['problem_type']} | metric: {info['metric']}\n"
          f"Training target summary: {info['training_target_summary']}\n"
          f"MLJAR: {version} ({supervised.__file__})\n"
          f"Device: {spec['device']} | threads: {spec['n_jobs']} | budget: {spec['seconds']}s\n", flush=True)
    X = pd.read_parquet(data / "X_train.parquet").reset_index(drop=True)
    y = pd.read_parquet(data / "y_train.parquet").iloc[:, 0].reset_index(drop=True)
    encoder = None
    if info["problem_type"] != "regression":
        encoder = LabelEncoder().fit(y)
        write_json(output / "classes.json", encoder.classes_.tolist())
        y = pd.Series(encoder.transform(y), name=y.name)
    settings = dict(mode="Compete", total_time_limit=spec["seconds"], n_jobs=spec["n_jobs"],
                    random_state=spec["seed"], explain_level=0, verbose=1,
                    results_path=str(output / "automl"),
                    algorithms=spec.get("algorithms") or "auto",
                    ml_task={"binary": "binary_classification", "multiclass": "multiclass_classification",
                             "regression": "regression"}[info["problem_type"]],
                    eval_metric={"roc_auc": "auc", "log_loss": "logloss", "rmse": "rmse"}[info["metric"]])
    if version in CANDIDATE_VERSIONS:
        settings["custom_params"] = {key: {"device": spec["device"]}
                                      for key in ["realmlp", "tabm", "kumo", "tabiclv2"]}
        for key in ["kumo", "tabiclv2"]:
            settings["custom_params"][key]["local_files_only"] = True
        if spec["device"] == "cpu" and not spec.get("algorithms"):
            settings["algorithms"] = AutoML(mode="Compete")._get_algorithms() + ["Kumo", "TabICLv2"]
    write_json(output / "requested_settings.json", settings)
    automl = AutoML(**settings)
    started = time.monotonic()
    write_json(output / "phase.json", dict(phase="training", started_monotonic=started))
    automl.fit(X, y)
    fit_seconds = time.monotonic() - started
    evaluation_started = time.monotonic()
    write_json(output / "phase.json", dict(phase="evaluation", started_monotonic=evaluation_started))
    params = read_json(output / "automl/params.json")
    write_json(output / "resolved_settings.json", params)
    leaderboard = pd.read_csv(output / "automl/leaderboard.csv")
    print("\nMLJAR leaderboard (internal validation):\n" + leaderboard.to_string(index=False), flush=True)
    inventory = model_inventory(automl, output / "automl", leaderboard)
    save_table(output / "model_inventory.csv", inventory)
    algorithm_coverage = coverage(output / "automl", inventory)
    write_json(output / "algorithm_coverage.json", algorithm_coverage)
    print("\nNew algorithms: " + ", ".join(f"{name}: {len(row['completed_models'])} models"
                                           for name, row in algorithm_coverage.items()), flush=True)

    # Outer test frames are first read after fit, never provided to tuning or selection.
    X_test = pd.read_parquet(data / "X_test.parquet")
    y_test = pd.read_parquet(data / "y_test.parquet").iloc[:, 0]
    encoded_test = encoder.transform(y_test) if encoder is not None else y_test.to_numpy()
    write_json(output / "test_row_indices.json", X_test.index.tolist())
    predictions_dir = output / "predictions"
    predictions_dir.mkdir()
    best_name = params["best_model"]
    prediction_started = time.monotonic()
    best_predictions = automl.predict_all(X_test)
    prediction_seconds = time.monotonic() - prediction_started
    score, error = score_predictions(best_predictions, encoded_test, info["problem_type"],
                                    len(encoder.classes_) if encoder is not None else None)
    best_predictions.to_parquet(predictions_dir / "best.parquet", index=False)
    metrics = dict(dataset=spec["dataset"], task_id=info["task_id"], version=version, fold=0, repeat=0,
                   split="r0f0", metric=info["metric"], test_score=score, metric_error=error,
                   best_model=best_name, fit_seconds=fit_seconds, n_models=len(inventory),
                   time_train_s=fit_seconds, time_infer_s=prediction_seconds,
                   device=spec["device"], n_jobs=spec["n_jobs"], seconds=spec["seconds"],
                   seed=spec["seed"], dataset_sha256=spec["dataset_sha256"])
    write_json(output / "metrics.json", metrics)
    print(f"\nHELD-OUT TEST: {info['metric']}={score:.8g}; selected by validation: {best_name}", flush=True)
    by_name = {model.get_name(): model for model in automl._models}
    for index, record in enumerate(inventory, 1):
        name = record["name"]
        tick = time.monotonic()
        print(f"Test predictions {index}/{len(inventory)}: {name}", flush=True)
        try:
            # Neither release exposes a public per-model predict API. Passing the fitted
            # model object preserves MLJAR's stacking/ensemble preprocessing path.
            frame = best_predictions if name == best_name else automl._base_predict(X_test, model=by_name[name])
            model_score, model_error = score_predictions(frame, encoded_test, info["problem_type"],
                                                         len(encoder.classes_) if encoder is not None else None)
            frame.to_parquet(predictions_dir / f"{name}.parquet", index=False)
            record.update(test_status="success", test_score=model_score, metric_error=model_error)
        except Exception as error:
            record.update(test_status="error", test_error=f"{type(error).__name__}: {error}")
            traceback.print_exc()
        record["test_prediction_seconds"] = time.monotonic() - tick
        save_table(output / "model_inventory.csv", inventory)
    print("Verifying that the selected saved model reloads...", flush=True)
    reloaded = AutoML(results_path=str(output / "automl"))
    actual = reloaded.predict_all(X_test.iloc[:20])
    pd.testing.assert_frame_equal(actual.reset_index(drop=True), best_predictions.iloc[:20].reset_index(drop=True),
                                  check_exact=False, rtol=1e-5, atol=1e-7)
    metrics.update(evaluation_seconds=time.monotonic() - evaluation_started, reload_verified=True,
                   per_model_prediction_failures=sum(r["test_status"] == "error" for r in inventory))
    write_json(output / "metrics.json", metrics)
    print(f"Saved all models and logs: {output}\nReport: {output / 'automl/README.md'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    args = parser.parse_args()
    main(read_json(args.case))
