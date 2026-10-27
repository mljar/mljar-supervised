"""Full-context fitting and fixed admission policies, without downloads."""
import os
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from supervised.algorithms.sdm import (
    KumoClassificationAlgorithm, TabICLv2ClassificationAlgorithm,
    KumoRegressionAlgorithm, TabICLv2RegressionAlgorithm,
    MAX_TRAINING_ROWS, validate_options,
)
from supervised.exceptions import ResourceAdmissionError

CLASSES = [KumoClassificationAlgorithm, TabICLv2ClassificationAlgorithm]


@pytest.mark.parametrize("cls", CLASSES)
def test_fixed_threshold_boundary_before_dependencies(cls, monkeypatch):
    model = cls({"ml_task": "binary_classification"})
    limit = MAX_TRAINING_ROWS[model.algorithm_short_name]
    model._check_training_size(limit)
    dependencies = Mock(side_effect=AssertionError("must not import backend"))
    monkeypatch.setattr("supervised.algorithms.sdm.check_dependencies", dependencies)
    X = np.zeros((limit + 1, 1))
    with pytest.raises(ResourceAdmissionError) as exc:
        model.fit(X, np.zeros(len(X)), X_validation=X[:2], y_validation=[0, 1])
    assert exc.value.resources == {
        "rows": limit + 1, "max_training_rows": limit, "reason": "training_rows"
    }
    dependencies.assert_not_called()


@pytest.mark.parametrize("algorithm", ["Kumo", "TabICLv2"])
def test_removed_and_nonconfigurable_options(algorithm):
    with pytest.raises(ValueError, match="no longer supported"):
        validate_options({"max_context_size": 4096}, algorithm)
    with pytest.raises(ValueError, match="Unknown"):
        validate_options({"max_rows": 100}, algorithm)


@pytest.mark.parametrize("cls", CLASSES)
def test_legacy_constructor_can_load_but_cannot_refit(cls):
    # Factory constructs saved wrappers with their old params before loading.
    model = cls({"ml_task": "binary_classification", "max_context_size": 1024})
    with pytest.raises(ValueError, match="no longer supported"):
        model.fit(np.zeros((4, 1)), [0, 1, 0, 1],
                  X_validation=np.ones((2, 1)), y_validation=[0, 1])


@pytest.mark.parametrize("cls", CLASSES + [KumoRegressionAlgorithm, TabICLv2RegressionAlgorithm])
def test_full_context_and_fold_isolation(cls, monkeypatch, tmp_path):
    pytest.importorskip("torch")
    regression = "Regression" in cls.__name__
    model = cls({"ml_task": "regression" if regression else "multiclass_classification",
                 "device": "cpu", "num_estimators": 1})
    rows = 4100
    # Duplicate/nonconsecutive indices must not drop or misalign any rows.
    X = pd.DataFrame({"value": np.arange(rows, dtype=float)}, index=np.arange(rows) // 2)
    X.iloc[0, 0] = np.nan
    y = np.arange(rows, dtype=float) if regression else np.array([0] * (rows - 2) + [1, 2])
    valid = pd.DataFrame({"value": [1e8, np.nan, -1e8]})
    checkpoint = tmp_path / "weights"
    checkpoint.write_bytes(b"fake")
    model.checkpoint_info = {}
    monkeypatch.setattr("supervised.algorithms.sdm.check_dependencies", lambda: None)
    monkeypatch.setattr("supervised.algorithms.sdm.version", lambda _: "0.2.1")
    monkeypatch.setattr(model, "_download", lambda: setattr(model, "checkpoint_path", str(checkpoint)))
    monkeypatch.setattr(model, "_load_network", lambda: setattr(model, "model", object()))
    resources = Mock(return_value={})
    monkeypatch.setattr(model, "_check_resources", resources)
    monkeypatch.setattr(model, "predict", lambda _: np.array([0., 1., 2.]) if regression else np.eye(3) * .8 + .2 / 3)
    model.fit(X, y, X_validation=valid, y_validation=[0, 1, 2])
    assert len(model.context) == rows
    np.testing.assert_array_equal(model.context_y, y)
    np.testing.assert_array_equal(model.context.f0.iloc[1:], X.value.iloc[1:])
    assert model.context.f0.iloc[0] == X.value.mean()
    assert model.training_info["context_rows"] == model.training_info["training_rows"] == rows
    assert all(call.args[0] == rows for call in resources.call_args_list)


@pytest.mark.parametrize("cls", CLASSES)
def test_memory_skip_before_preprocessing_and_download(cls, monkeypatch):
    pytest.importorskip("torch")
    model = cls({"ml_task": "binary_classification", "device": "cpu",
                 "memory_limit_gb": .001})
    monkeypatch.setattr("supervised.algorithms.sdm.check_dependencies", lambda: None)
    preprocess, download = Mock(), Mock()
    monkeypatch.setattr(model, "_preprocess", preprocess)
    monkeypatch.setattr(model, "_download", download)
    X = np.arange(4100).reshape(-1, 1)
    with pytest.raises(ResourceAdmissionError, match="RAM exceeds") as exc:
        model.fit(X, np.arange(4100) % 2, X_validation=X[:2], y_validation=[0, 1])
    assert exc.value.resources["rows"] == 4100
    preprocess.assert_not_called()
    download.assert_not_called()


@pytest.mark.skipif(
    os.environ.get("MLJAR_TEST_SDM") != "1",
    reason="opt-in checkpoint integration",
)
@pytest.mark.parametrize("cls", CLASSES)
def test_real_full_context_above_old_cap_and_legacy_reload(cls, tmp_path):
    import os
    from sklearn.datasets import make_classification
    from supervised.algorithms.factory import AlgorithmFactory

    X, y = make_classification(n_samples=4120, n_features=4, n_redundant=0,
                               random_state=42)
    model = cls({"ml_task": "binary_classification", "model_type": cls.algorithm_short_name,
                 "device": os.environ.get("MLJAR_TEST_SDM_DEVICE", "cpu"),
                 "num_estimators": 1, "predict_batch_size": 16,
                 "n_jobs": 4, "local_files_only": True})
    model.fit(X[:4100], y[:4100], X_validation=X[4100:], y_validation=y[4100:], max_time=180)
    assert model.training_info["context_rows"] == 4100
    np.testing.assert_array_equal(model.context_y, y[:4100])
    prediction = model.predict(X[4100:])
    assert np.isfinite(prediction).all()
    # Legacy payloads contain the now-removed option. Loading must preserve
    # their saved context even if it exceeds the value of that old option.
    model.params["max_context_size"] = 4096
    path = str(tmp_path / "fold.model")
    model.save(path)
    description = model.get_params()
    model.unload()
    restored = AlgorithmFactory.load(description, path, lazy_load=False)
    assert len(restored.context) == 4100
    np.testing.assert_allclose(restored.predict(X[4100:]), prediction, atol=1e-6)
    restored.unload()


def test_kumo_effective_feature_limit_skips_before_download(monkeypatch):
    pytest.importorskip("torch")
    model = KumoClassificationAlgorithm({"ml_task": "binary_classification", "device": "cpu"})
    monkeypatch.setattr("supervised.algorithms.sdm.check_dependencies", lambda: None)
    monkeypatch.setattr(model, "_check_resources", lambda *args: {})
    download = Mock()
    monkeypatch.setattr(model, "_download", download)
    # A numeric missing indicator takes 500 raw columns over the effective cap.
    X = pd.DataFrame(np.arange(2000, dtype=float).reshape(4, 500))
    X.iloc[0, 0] = np.nan
    with pytest.raises(ResourceAdmissionError, match="500 effective columns") as exc:
        model.fit(X, [0, 1, 0, 1], X_validation=X[:2], y_validation=[0, 1])
    assert exc.value.resources["features"] == 501
    download.assert_not_called()
