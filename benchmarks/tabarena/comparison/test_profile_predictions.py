from types import SimpleNamespace

import pytest

from profile_predictions import PredictionProfiler, validate_reference, main
from common import sha256, write_json, read_json


def test_profiler_records_failures_and_restores_methods():
    class Model:
        def predict(self):
            raise ValueError("prediction failed")

    original = Model.predict
    snapshots = iter([{"allocated_bytes": 100}, {"allocated_bytes": 0}])
    profiler = PredictionProfiler(memory_snapshot=lambda: next(snapshots))
    try:
        profiler.instrument(Model, "predict", "query")
        with pytest.raises(ValueError, match="prediction failed"):
            Model().predict()
        assert len(profiler.events) == 1
        assert profiler.events[0]["success"] is False
        assert profiler.events[0]["seconds"] >= 0
        assert profiler.events[0]["memory_before"]["allocated_bytes"] == 100
        assert profiler.events[0]["memory_after"]["allocated_bytes"] == 0
    finally:
        profiler.stack.close()
    assert Model.predict is original


def test_profiler_distinguishes_fast_memory_probes_from_cleanup():
    class Resources:
        @staticmethod
        def available_gpu_memory(device, *, cleanup=True):
            return 1, 2

    profiler = PredictionProfiler()
    try:
        profiler.instrument(Resources, "available_gpu_memory", "gpu_admission_cleanup")
        Resources.available_gpu_memory("cuda:0", cleanup=False)
        Resources.available_gpu_memory("cuda:0", cleanup=True)
        Resources.available_gpu_memory("cuda:0")
        assert [event["cleanup"] for event in profiler.events] == [False, True, True]
    finally:
        profiler.stack.close()


@pytest.mark.parametrize(
    "field",
    [
        "artifact_sha256",
        "dataset_sha256",
        "device",
        "n_jobs",
        "cpu_affinity",
        "packages",
        "rows",
        "repeats",
        "status",
    ],
)
def test_reference_requires_matching_protocol(field):
    reference = dict(
        artifact_sha256={},
        dataset_sha256="data",
        device="cpu",
        n_jobs=2,
        cpu_affinity=[0, 1],
        packages={},
        rows=[1, 20],
        repeats=2,
        status="success",
    )
    validate_reference(reference, reference)
    changed = dict(reference, **{field: "changed"})
    with pytest.raises(ValueError):
        validate_reference(reference, changed)


def test_profile_saved_model_and_reference_end_to_end(tmp_path, monkeypatch):
    pytest.importorskip("sdm")
    import torch
    import pandas as pd
    from sklearn.datasets import make_classification
    from supervised import AutoML
    from supervised.model_framework import ModelFramework

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    X, y = make_classification(n_samples=80, n_features=4, random_state=42)
    X = pd.DataFrame(X, columns=list("abcd"))
    attempt, data = tmp_path / "attempt", tmp_path / "data"
    attempt.mkdir()
    model = AutoML(
        results_path=str(attempt / "automl"),
        algorithms=["Decision Tree"],
        validation_strategy={"validation_type": "split", "train_ratio": 0.8},
        explain_level=0,
        train_ensemble=False,
        verbose=0,
        n_jobs=1,
    )
    model.fit(X.iloc[:60], y[:60])
    data.mkdir()
    X.iloc[60:].to_parquet(data / "X_test.parquet")
    write_json(
        data / "dataset.json",
        dict(files_sha256={"X_test.parquet": sha256(data / "X_test.parquet")}),
    )
    write_json(
        attempt / "case.json",
        dict(
            data=str(data), version="test", dataset_sha256=sha256(data / "dataset.json")
        ),
    )
    args = SimpleNamespace(
        attempt=attempt,
        output=tmp_path / "baseline",
        source_root=None,
        reference=None,
        device="cpu",
        n_jobs=1,
        rows=[1, 20],
        repeats=2,
    )
    original = ModelFramework.predict
    main(args)
    assert ModelFramework.predict is original
    manifest = read_json(args.output / "manifest.json")
    assert manifest["status"] == "success" and manifest["reload_verified"]
    assert manifest["artifacts_unchanged"]
    assert len(read_json(args.output / "timings.json")) == 6
    args.reference, args.output = args.output, tmp_path / "patched"
    main(args)
    assert read_json(args.output / "manifest.json")["status"] == "success"
