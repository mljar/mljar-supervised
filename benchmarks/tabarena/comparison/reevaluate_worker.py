"""Predict from saved artifacts using verified frozen inference code."""

from __future__ import annotations

import argparse
from collections import Counter
import gc
import importlib.metadata
from pathlib import Path
import re
import shutil
import sys
import time

from common import read_json, sha256, write_json
from reevaluate import tree_hashes, verify_hashes


def runtime_packages():
    return {
        re.sub(r"[-_.]+", "-", d.metadata["Name"]).lower(): d.version
        for d in importlib.metadata.distributions()
    }


def encode_labels(y, classes):
    import pandas as pd

    mapping = pd.Index(classes)
    if not mapping.is_unique:
        raise ValueError("Saved class mapping contains duplicates")
    codes = mapping.get_indexer(y)
    if (codes < 0).any():
        raise ValueError(
            "Held-out labels are absent from the saved training class mapping"
        )
    return codes


def gpu_stats(torch, device):
    if device != "cuda":
        return {}
    torch.cuda.synchronize()
    return dict(
        device=torch.cuda.get_device_name(),
        peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3,
        peak_reserved_gib=torch.cuda.max_memory_reserved() / 1024**3,
        end_allocated_gib=torch.cuda.memory_allocated() / 1024**3,
        end_reserved_gib=torch.cuda.memory_reserved() / 1024**3,
    )


def assert_unloaded(automl, predicted=()):
    resident = [
        model.get_name()
        for model in automl._models
        for learner in getattr(model, "learners", [])
        if learner.is_fitted()
        or (model.get_name() in predicted and learner.model is not None)
    ]
    if resident:
        raise RuntimeError(f"Saved fold learners remain resident: {resident}")


def main(spec):
    snapshot, source, output = map(
        Path, (spec["snapshot"], spec["source_attempt"], spec["attempt"])
    )
    started = time.monotonic()
    write_json(
        output / "phase.json", dict(phase="evaluation", started_monotonic=started)
    )
    manifest = read_json(snapshot / "manifest.json")
    verify_hashes(snapshot / "source", manifest["source_sha256"])
    verify_hashes(snapshot / "code", manifest["runner_sha256"])
    environment = read_json(snapshot / "environment.json")
    expected = {
        re.sub(r"[-_.]+", "-", key).lower(): value
        for key, value in environment["packages"].items()
    }
    actual = runtime_packages()
    if actual != expected:
        changed = {
            key: [expected.get(key), actual.get(key)]
            for key in set(actual) | set(expected)
            if expected.get(key) != actual.get(key)
        }
        raise RuntimeError(f"Original dependency environment changed: {changed}")
    # Import only after selecting the frozen source, including uncommitted fixes.
    sys.path.insert(0, str(snapshot / "source"))
    import pandas as pd
    import torch
    import supervised
    from supervised import AutoML
    from supervised.model_framework import ModelFramework
    from worker import score_predictions

    if (
        not Path(supervised.__file__)
        .resolve()
        .is_relative_to(snapshot / "source/supervised")
    ):
        raise RuntimeError(f"Wrong inference source: {supervised.__file__}")
    if supervised.__version__ != spec["training_version"]:
        raise RuntimeError(f"Unexpected source version: {supervised.__version__}")
    if spec["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Original fit used CUDA but CUDA is unavailable")
    torch.set_num_threads(spec["n_jobs"])
    write_json(
        output / "runtime.json",
        dict(
            python=sys.executable,
            packages=actual,
            supervised_file=supervised.__file__,
            source_digest=manifest["source_digest"],
            label=spec["label"],
            training_version=spec["training_version"],
        ),
    )

    data = Path(spec["data"])
    if sha256(data / "dataset.json") != spec["dataset_sha256"]:
        raise RuntimeError("Dataset manifest changed")
    info = read_json(data / "dataset.json")
    for name, digest in info["files_sha256"].items():
        if sha256(data / name) != digest:
            raise RuntimeError(f"Dataset file changed: {name}")
    print(
        f"\n{spec['label']} | {info['dataset']} | {info['domain']}\n{info['description']}\n"
        f"Official {info['split']}: {info['train_rows']:,} training / {info['test_rows']:,} test rows; "
        f"{info['features']} features; metric {info['metric']}\n"
        f"Saved models: {source / 'automl'}\nInference code: {supervised.__file__}\n"
        f"Device: {spec['device']}; threads: {spec['n_jobs']}\n",
        flush=True,
    )
    print("Fingerprinting original artifacts...", flush=True)
    source_hashes = tree_hashes(source)
    write_json(output / "source_artifacts_sha256.json", source_hashes)
    for name in (
        "dataset.json",
        "classes.json",
        "model_inventory.csv",
        "algorithm_coverage.json",
    ):
        if (source / name).exists():
            shutil.copy2(source / name, output / name)
    params = read_json(source / "automl/params.json")
    X = pd.read_parquet(data / "X_test.parquet")
    y = pd.read_parquet(data / "y_test.parquet").iloc[:, 0]
    if len(X) != info["test_rows"] or len(y) != len(X) or not X.index.equals(y.index):
        raise ValueError("Held-out rows or indices do not match the original split")
    classes = (
        read_json(source / "classes.json")
        if info["problem_type"] != "regression"
        else None
    )
    encoded = encode_labels(y, classes) if classes is not None else y.to_numpy()
    write_json(output / "test_row_indices.json", X.index.tolist())
    (output / "predictions").mkdir()
    calls, trace = Counter(), []
    original_predict = ModelFramework.predict

    def logged_predict(model, features):
        name = model.get_name()
        calls[name] += 1
        tick = time.monotonic()
        print(f"Predict {name} ({len(features):,} rows)", flush=True)
        result = original_predict(model, features)
        duration = time.monotonic() - tick
        trace.append(
            dict(
                model=name,
                algorithm=model.get_type(),
                rows=len(features),
                seconds=duration,
            )
        )
        print(f"Done {name}: {duration:.2f}s", flush=True)
        return result

    ModelFramework.predict = logged_predict
    try:
        tick = time.monotonic()
        automl = AutoML(results_path=str(source / "automl"))
        automl.load(str(source / "automl"))
        load_seconds = time.monotonic() - tick
        assert_unloaded(automl)
        print(
            f"Loaded metadata in {load_seconds:.2f}s; selected model: {params['best_model']}",
            flush=True,
        )
        if spec["device"] == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        tick = time.monotonic()
        predictions = automl.predict_all(X)
        gpu = gpu_stats(torch, spec["device"])
        prediction_seconds = time.monotonic() - tick
        assert_unloaded(automl, calls)
        if any(count != 1 for count in calls.values()):
            raise RuntimeError(
                f"A model was evaluated repeatedly within one prediction: {dict(calls)}"
            )
        score, error = score_predictions(
            predictions,
            encoded,
            info["problem_type"],
            len(classes) if classes is not None else None,
        )
        predictions.to_parquet(output / "predictions/best.parquet", index=False)
        metrics = dict(
            dataset=spec["dataset"],
            task_id=info["task_id"],
            label=spec["label"],
            version=spec["label"],
            training_version=spec["training_version"],
            source_digest=manifest["source_digest"],
            split=info["split"],
            repeat=0,
            fold=0,
            metric=info["metric"],
            test_score=score,
            metric_error=error,
            best_model=params["best_model"],
            n_models=len(params["saved"]),
            dependency_models=len(calls),
            dependency_calls=dict(calls),
            load_seconds=load_seconds,
            time_infer_s=prediction_seconds,
            gpu=gpu,
            n_jobs=spec["n_jobs"],
            device=spec["device"],
            seed=spec["seed"],
            seconds=spec["seconds"],
            dataset_sha256=spec["dataset_sha256"],
            reload_verified=False,
            original_artifacts_unchanged=False,
        )
        write_json(output / "metrics.json", metrics)
        write_json(output / "dependency_predictions.json", trace)
        print(
            f"HELD-OUT TEST: {info['metric']}={score:.10g}; prediction {prediction_seconds:.2f}s; "
            f"GPU {gpu}",
            flush=True,
        )
        old_predictions = source / "predictions/best.parquet"
        if old_predictions.exists():
            pd.testing.assert_frame_equal(
                predictions.reset_index(drop=True),
                pd.read_parquet(old_predictions),
                check_exact=False,
                rtol=1e-4,
                atol=1e-6,
            )
            metrics["original_prediction_parity"] = True
        # GPU kernels may round differently for different batch shapes. Compare
        # identical batches across reloads, not a small batch against the full set.
        print("Creating the 20-row reference for reload verification...", flush=True)
        calls.clear()
        trace.clear()
        check_X = X.iloc[:20].copy()
        tick = time.monotonic()
        reference = automl.predict_all(check_X)
        assert_unloaded(automl, calls)
        metrics.update(
            reload_reference_seconds=time.monotonic() - tick,
            reload_check_rows=len(check_X),
        )
        write_json(output / "reload_reference_predictions.json", trace)
        del automl
        gc.collect()
        calls.clear()
        trace.clear()
        print("Checking a fresh AutoML reload on the first 20 test rows...", flush=True)
        tick = time.monotonic()
        reloaded = AutoML(results_path=str(source / "automl"))
        reloaded.load(str(source / "automl"))
        assert_unloaded(reloaded)
        check = reloaded.predict_all(check_X.copy())
        pd.testing.assert_frame_equal(
            check.reset_index(drop=True),
            reference.reset_index(drop=True),
            check_exact=False,
            rtol=1e-5,
            atol=1e-7,
        )
        assert_unloaded(reloaded, calls)
        write_json(output / "reload_dependency_predictions.json", trace)
        metrics.update(
            reload_verified=True,
            reload_seconds=time.monotonic() - tick,
            gpu_including_reload=gpu_stats(torch, spec["device"]),
        )
        print("Checking original artifacts are unchanged...", flush=True)
        verify_hashes(source, source_hashes)
        metrics.update(
            original_artifacts_unchanged=True,
            evaluation_seconds=time.monotonic() - started,
        )
        write_json(output / "metrics.json", metrics)
        print(
            f"PASS {spec['dataset']}: full test set, fresh reload, fold cleanup, original artifact checks.",
            flush=True,
        )
    finally:
        ModelFramework.predict = original_predict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    main(read_json(parser.parse_args().case))
