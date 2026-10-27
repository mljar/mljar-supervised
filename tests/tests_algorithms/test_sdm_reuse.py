"""Saved-fold equivalence and bounded frozen-weight reuse, without downloads."""

import os
import weakref

import joblib
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sdm")
from supervised.algorithms.sdm import KumoClassificationAlgorithm
from supervised.prediction import PredictionContext, current_prediction_context
from tests.tests_algorithms.test_sdm_context import Network, learner


@pytest.fixture
def saved(learner, tmp_path, monkeypatch):
    checkpoint = tmp_path / "classifier.pt"
    checkpoint.write_bytes(b"fake frozen weights")
    learner.checkpoint_path = str(checkpoint)
    learner.checkpoint_info = {"revision": "test"}
    learner.notice_paths = []
    learner.save(str(tmp_path / "first.kumo"))
    loads, live = [], weakref.WeakSet()

    def load(*args, **kwargs):
        assert not live, "Two frozen networks were retained"
        network = Network()
        loads.append(weakref.ref(network))
        live.add(network)
        return network

    monkeypatch.setattr("supervised.algorithms._sdm_backend.load_network", load)

    def make(**changes):
        payload = joblib.load(tmp_path / "first.kumo")
        payload["params"].update(changes)
        path = tmp_path / f"candidate-{len(list(tmp_path.glob('candidate-*')))}.kumo"
        joblib.dump(payload, path)
        model = KumoClassificationAlgorithm({"ml_task": "binary_classification"})
        model.model_file_path = str(path)
        return model

    return make, loads, live


def predict_unload(model, X):
    try:
        return model.predict(X)
    finally:
        model.unload()


def test_equivalent_caps_reuse_predictions_but_preserve_request_lifetime(saved):
    make, loads, live = saved
    X = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
    models = [make(max_context_size=cap) for cap in (1024, 4096, 8192)]
    for _ in range(2):
        with PredictionContext() as request:
            expected = predict_unload(models[0], X)
            for model in models[1:]:
                result = predict_unload(model, X.copy())
                np.testing.assert_array_equal(result, expected)
                result[:] = -100  # A caller cannot mutate the cached prediction.
            assert loads[-1]().fits == 1
            assert loads[-1]()._cache is None
            assert len(live) == 1
        assert not live
        assert request._sdm_prediction_bytes == 0
        assert not request._sdm_predictions
    assert len(loads) == 2


@pytest.mark.parametrize(
    "change",
    [
        {"seed": 99},
        {"num_estimators": 2},
        {"precision": "fp32"},
        {"predict_batch_size": 5},
    ],
)
def test_different_prediction_settings_share_weights_only(saved, change):
    make, loads, live = saved
    X = pd.DataFrame({"a": [1.0, 2.0]})
    with PredictionContext():
        predict_unload(make(), X)
        predict_unload(make(**change), X)
        assert len(loads) == 1
        assert loads[0]().fits == 2
        assert loads[0]()._cache is None
    assert not live


@pytest.mark.parametrize(
    "field",
    [
        "context",
        "context_y",
        "classes_",
        "category_maps",
        "numeric_means",
        "indicators",
        "columns",
        "checkpoint",
        "query",
        "query_order",
    ],
)
def test_fingerprint_includes_fitted_state_and_input(saved, field, tmp_path):
    make, _, _ = saved
    model = make()
    model._load_state(model.model_file_path)
    X = pd.DataFrame({"a": [1.0, 2.0]})
    before = model._prediction_key(X)
    if field == "checkpoint":
        from pathlib import Path

        Path(model.checkpoint_path).write_bytes(b"replacement")
    elif field == "query":
        X.iloc[0, 0] = 5
    elif field == "query_order":
        X = X.iloc[::-1]
    elif field == "context":
        model.context = model.context.iloc[::-1]
    elif field in ["context_y", "classes_"]:
        setattr(model, field, getattr(model, field)[::-1])
    elif field in ["category_maps", "numeric_means"]:
        getattr(model, field)["new"] = 1
    else:
        getattr(model, field).append("new")
    assert before != model._prediction_key(X)


def test_changed_query_and_fold_context_are_recomputed(saved):
    make, loads, _ = saved
    model = make()
    payload = joblib.load(model.model_file_path)
    payload["context_y"] = payload["context_y"][::-1]
    other = make()
    joblib.dump(payload, other.model_file_path)
    X = pd.DataFrame({"a": [1.0, 2.0]})
    with PredictionContext():
        first = predict_unload(model, X)
        predict_unload(other, X)
        X.iloc[0, 0] = 3
        changed = predict_unload(model, X)
        assert first[0] != changed[0]
        assert loads[0]().fits == 3


def test_checkpoint_switch_evicts_previous_network(saved):
    make, loads, live = saved
    X = pd.DataFrame({"a": [1.0]})
    with PredictionContext():
        predict_unload(make(size="small"), X)
        predict_unload(make(size="medium"), X)
        assert len(loads) == 2 and len(live) == 1
        assert loads[0]() is None
    assert not live


def test_stacking_keeps_equivalent_columns_and_releases_weights_for_other_algorithms(
    saved,
):
    from types import SimpleNamespace
    from supervised import AutoML
    from supervised.model_framework import ModelFramework
    from tests.tests_automl.test_prediction_lifecycle import framework

    make, loads, live = saved
    models = []
    for name, cap in [("1_Kumo", 1024), ("2_Kumo", 4096)]:
        m = ModelFramework(
            {"learner": {"model_type": "Kumo"}, "validation_strategy": {}}
        )
        m._name = name
        m.learners = [make(max_context_size=cap, seed=i) for i in (1, 2)]
        m.preprocessings = [
            SimpleNamespace(
                transform=lambda X, y: (X, y, None),
                inverse_scale_target=lambda p: p,
                prepare_target_labels=lambda p: pd.DataFrame(
                    {"prediction_0": 1 - p, "prediction_1": p}
                ),
            )
        ] * 2
        models.append(m)
    automl = AutoML()
    automl._ml_task = "binary_classification"
    automl._stacked_models = models
    X = pd.DataFrame({"a": [1.0, 2.0]}, index=[9, 10])
    with PredictionContext() as request:
        stacked = automl.get_stacked_data(X, mode="predict", prediction_context=request)
        assert stacked.columns.tolist() == [
            "a",
            "1_Kumo_prediction",
            "2_Kumo_prediction",
        ]
        assert stacked.index.tolist() == [9, 10]
        np.testing.assert_array_equal(stacked.iloc[:, 1], stacked.iloc[:, 2])
        assert loads[0]().fits == 2  # Four saved folds, two distinct contexts/seeds.
        assert all(l.model is None for m in models for l in m.learners)
        other, _ = framework()
        other.predict(pd.DataFrame({"feature": [1.0]}))
        assert not live


def test_resource_checks_still_apply_and_evict_before_retry(saved, monkeypatch):
    from supervised.exceptions import AutoMLException

    make, loads, live = saved
    checks = []

    def check(*args):
        checks.append(len(live))
        if live:
            raise AutoMLException("budget")

    monkeypatch.setattr(KumoClassificationAlgorithm, "_check_resources", check)
    with PredictionContext():
        for seed in (1, 2):
            predict_unload(make(seed=seed), pd.DataFrame({"a": [1.0]}))
        assert checks == [0, 1, 0]
        assert len(loads) == 2
    assert not live


def test_actual_admission_evicts_retained_weights_after_cleanup_fails(
    saved, monkeypatch
):
    make, loads, live = saved
    probes = []

    def available(device, *, cleanup=True):
        probes.append((cleanup, len(live)))
        return (0 if live else 16 * 1024**3), 16 * 1024**3

    monkeypatch.setattr(
        "supervised.algorithms.realmlp_resources.available_gpu_memory", available
    )
    monkeypatch.setattr(
        "supervised.algorithms.realmlp_resources.gpu_allocation_stats", lambda device: {}
    )
    with PredictionContext() as request:
        for _ in range(2):
            model = make()
            model._load_state(model.model_file_path)
            model.device = "cuda:1"  # The test loader uses CPU fake networks.
            model._restore_network(request)
            model.unload()
        assert probes == [(False, 0), (False, 1), (True, 1), (False, 0)]
        assert len(loads) == 2
    assert not live


def test_failure_clears_request_and_allows_retry(saved, monkeypatch):
    make, loads, live = saved
    model = make()
    original = Network.predict

    def fail(*args, **kwargs):
        raise RuntimeError("query failed")

    monkeypatch.setattr(Network, "predict", fail)
    with pytest.raises(RuntimeError, match="query failed"):
        with PredictionContext() as request:
            predict_unload(model, pd.DataFrame({"a": [1.0]}))
    assert not live and not request._sdm_predictions
    assert current_prediction_context() is None
    monkeypatch.setattr(Network, "predict", original)
    with PredictionContext():
        predict_unload(model, pd.DataFrame({"a": [1.0]}))
    assert len(loads) == 2 and not live


def test_cpu_prediction_cache_budget_and_nested_scopes():
    with PredictionContext() as outer:
        outer.keep_sdm_prediction("a", np.ones(2), 32)
        outer.keep_sdm_prediction("b", np.ones(2), 32)
        assert outer.sdm_prediction("a") is not None  # LRU refresh
        outer.keep_sdm_prediction("c", np.ones(2), 32)
        assert outer.sdm_prediction("b") is None
        outer.keep_sdm_prediction("large", np.ones(10), 32)
        assert outer.sdm_prediction("large") is None
        assert outer._sdm_prediction_bytes <= 32
        with PredictionContext() as inner:
            assert current_prediction_context() is inner
            assert inner.sdm_prediction("a") is None
        assert current_prediction_context() is outer
    assert current_prediction_context() is None


@pytest.mark.skipif(
    os.environ.get("MLJAR_TEST_SDM") != "1", reason="opt-in real checkpoints"
)
@pytest.mark.parametrize("algorithm", ["Kumo", "TabICLv2"])
@pytest.mark.parametrize(
    "task", ["binary_classification", "multiclass_classification", "regression"]
)
def test_real_checkpoint_reuse_parity(algorithm, task, tmp_path, monkeypatch):
    from sklearn.datasets import make_classification, make_regression
    from supervised.algorithms import sdm

    regression = task == "regression"
    cls = getattr(
        sdm,
        algorithm
        + ("RegressionAlgorithm" if regression else "ClassificationAlgorithm"),
    )
    if regression:
        X, y = make_regression(n_samples=100, n_features=4, random_state=42)
    else:
        X, y = make_classification(
            n_samples=100,
            n_features=4,
            n_informative=3,
            n_redundant=0,
            n_classes=3 if task == "multiclass_classification" else 2,
            random_state=42,
        )
    X = pd.DataFrame(X)
    X["category"] = np.where(X[0] > 0, "a", "b")
    X.loc[::7, 1] = np.nan
    model = cls(
        dict(
            ml_task=task,
            device=os.environ.get("MLJAR_TEST_SDM_DEVICE", "cpu"),
            num_estimators=2,
            predict_batch_size=8,
            n_jobs=2,
            local_files_only=True,
        )
    )
    model.fit(
        X.iloc[:80], y[:80], X_validation=X.iloc[80:], y_validation=y[80:], max_time=120
    )
    path = tmp_path / "fold.model"
    model.save(str(path))
    model.unload()
    other = tmp_path / "other.model"
    payload = joblib.load(path)
    payload["context_y"] = payload["context_y"][::-1].copy()
    joblib.dump(payload, other)
    query = X.iloc[80:89]

    def lazy(path):
        result = cls({"ml_task": task})
        result.model_file_path = str(path)
        return result

    references = [predict_unload(lazy(p), query) for p in (path, other)]
    load = cls._load_network
    calls = []

    def count(self):
        calls.append(True)
        return load(self)

    monkeypatch.setattr(cls, "_load_network", count)
    for _ in range(2):  # A fresh request must reconstruct its own network/context.
        with PredictionContext():
            for p, expected in zip((path, other, path), references + references[:1]):
                actual = predict_unload(lazy(p), query.copy())
                np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-7)
    assert len(calls) == 2
