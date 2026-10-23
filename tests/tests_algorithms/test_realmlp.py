import copy
import importlib.util
import json
import warnings
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from scipy.sparse import csr_matrix
from sklearn.datasets import make_classification, make_friedman1, make_regression
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.model_selection import train_test_split

from supervised.algorithms.realmlp import (
    RealMLPClassificationAlgorithm,
    RealMLPRegressionAlgorithm,
    SEARCH_PARAMS,
    check_dependencies,
    validate_options,
)
from supervised.algorithms.realmlp_resources import RealMLPResourceError, thread_count
from supervised.algorithms.registry import AlgorithmsRegistry
from supervised.exceptions import AutoMLException
from supervised.tuner.random_parameters import RandomParameters


has_backend = (
    importlib.util.find_spec("pytabkit") is not None
    and importlib.util.find_spec("torch") is not None
)
backend = pytest.mark.skipif(
    not has_backend, reason="install mljar-supervised[realmlp]"
)


def learner(task="binary_classification", **kwargs):
    cls = (
        RealMLPRegressionAlgorithm
        if task == "regression"
        else RealMLPClassificationAlgorithm
    )
    return cls(
        dict(ml_task=task, device="cpu", n_jobs=2, n_epochs=8, seed=42, **kwargs)
    )


def test_parameters_and_weight_capability():
    for seed in range(10):
        p = RandomParameters.get(SEARCH_PARAMS, seed)
        assert p["hidden_width"] in [32, 64, 128]
        assert p["n_hidden_layers"] in [1, 2, 3]
    assert not AlgorithmsRegistry.supports_sample_weight("regression", "RealMLP")
    assert AlgorithmsRegistry.supports_sample_weight("regression", "LightGBM")
    for params in [
        {"hidden_width": 512},
        {"n_hidden_layers": 4},
        {"foo": 1},
        {"memory_fraction": 0},
        {"device": "mps"},
        {"n_epochs": -1},
    ]:
        with pytest.raises(ValueError):
            validate_options(params)


def test_missing_dependencies():
    with patch(
        "supervised.algorithms.realmlp.importlib.util.find_spec", return_value=None
    ):
        with pytest.raises(AutoMLException, match=r"mljar-supervised\[realmlp\]"):
            check_dependencies()


def test_reject_weights_before_backend_import():
    for kwargs in [{"sample_weight": [1, 2]}, {"sample_weight_validation": [1, 2]}]:
        with pytest.raises(ValueError, match="weights"):
            learner().fit([[1], [2]], [0, 1], **kwargs)


def test_preprocessing_train_only_and_schema():
    m = learner()
    train = pd.DataFrame(
        {
            "a": [1.0, np.nan, 3.0],
            "b": ["yes", "no", None],
            "c": pd.Series([True, False, None], dtype="boolean"),
            "empty": [np.nan] * 3,
        }
    )
    before = train.copy(deep=True)
    transformed = m._preprocess(train, fit=True)
    valid = pd.DataFrame(
        {"empty": [np.nan], "c": [True], "b": ["unseen"], "a": [np.nan]}
    )
    out = m._preprocess(valid)
    pd.testing.assert_frame_equal(train, before)
    assert out["f0"].iloc[0] == 2
    assert out["f1"].iloc[0] == 1
    assert "unseen" not in m.category_maps["b"]
    assert transformed["missing0"].iloc[1] == 1
    assert out["f3"].iloc[0] == 0
    with pytest.raises(ValueError, match="columns"):
        m._preprocess(valid.drop(columns="a"))
    with pytest.raises(ValueError, match="sparse"):
        m._frame(csr_matrix([[1, 0], [0, 1]]))
    with pytest.raises(ValueError, match="sparse"):
        m._frame(pd.DataFrame.sparse.from_spmatrix(csr_matrix([[1, 0], [0, 1]])))


@backend
@pytest.mark.parametrize(
    "task", ["binary_classification", "multiclass_classification", "regression"]
)
def test_fit_reload_copy_and_validation_history(task, tmp_path):
    if task == "regression":
        X, y = make_regression(n_samples=400, n_features=10, random_state=42)
    else:
        X, y = make_classification(
            n_samples=400,
            n_features=10,
            n_informative=6,
            n_classes=3 if task.startswith("multi") else 2,
            random_state=42,
        )
    X, y = pd.DataFrame(X), pd.Series(y)
    tr, va, yt, yv = train_test_split(X, y, test_size=0.25, random_state=42)
    m = learner(task)
    import torch

    old_threads = torch.get_num_threads()
    m.fit(
        tr,
        yt,
        X_validation=va,
        y_validation=yv,
        log_to_file=str(tmp_path / "curve.log"),
        max_time=60,
    )
    assert torch.get_num_threads() == old_threads
    predictions = m.predict(va)
    assert np.isfinite(predictions).all()
    assert m.training_info["completed_epochs"] == 8
    assert 1 <= m.training_info["best_epoch"] <= 8
    if task == "binary_classification":
        assert (
            abs(
                log_loss(yv, predictions)
                - min(h["validation"] for h in m.training_info["history"])
            )
            < 1e-5
        )
    if task.startswith("multi"):
        assert predictions.shape == (100, 3)
        assert m.predict_proba(va).shape == (100, 3)
        assert_allclose(predictions.sum(axis=1), 1, atol=1e-6)
    else:
        assert predictions.shape == (100,)
    path = tmp_path / "model.realmlp"
    m.save(path)
    loaded = learner(task)
    loaded.load(path)
    assert loaded.device == "cpu"
    assert_allclose(predictions, loaded.predict(va), atol=1e-6)
    assert_allclose(predictions, m.copy().predict(va), atol=1e-6)
    assert len(pd.read_csv(tmp_path / "curve.log", header=None)) == 8


@backend
def test_mixed_data_and_chunked_prediction(tmp_path):
    X, y = make_classification(n_samples=250, n_features=8, random_state=42)
    X = pd.DataFrame(X)
    X["cat"] = pd.cut(X[0], bins=[-np.inf, 0, np.inf], labels=["low", "high"])
    X["bool"] = X[1] > 0
    X.loc[::7, 2] = np.nan
    X["empty"] = np.nan
    m = learner(predict_batch_size=17)
    m.fit(X.iloc[:200], y[:200], X_validation=X.iloc[200:], y_validation=y[200:])
    X = X.iloc[200:].copy()
    X["cat"] = "unknown"
    expected = m.predict(X)
    assert np.isfinite(expected).all()
    m.save(tmp_path / "mixed")
    m.load(tmp_path / "mixed")
    assert_allclose(expected, m.predict(X[list(reversed(X.columns))]), atol=1e-6)


@backend
def test_memory_guard_and_expired_budget():
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    with patch(
        "supervised.algorithms.realmlp_resources.available_memory", return_value=1024
    ), patch("supervised.algorithms._realmlp_backend.RealMLPClassifier.fit") as fit:
        with pytest.raises(RealMLPResourceError):
            learner().fit(X, y)
        fit.assert_not_called()
    with pytest.raises(AutoMLException, match="time budget"):
        learner().fit(X, y, max_time=0)


@backend
def test_early_stopping_and_no_validation():
    X, y = make_classification(n_samples=200, n_features=10, random_state=42)
    m = learner(
        use_early_stopping=True,
        early_stopping_additive_patience=0,
        early_stopping_multiplicative_patience=0,
    )
    # A deterministic immediate stop before a checkpoint must fail, not save random weights.
    with pytest.raises(AutoMLException, match="validation epoch"):
        m.fit(X[:150], y[:150], X_validation=X[150:], y_validation=y[150:])
    m = learner(
        use_early_stopping=True,
        early_stopping_additive_patience=2,
        early_stopping_multiplicative_patience=0,
    )
    m.fit(X[:150], y[:150], X_validation=X[150:], y_validation=y[150:])
    assert m.training_info["completed_epochs"] == 3
    assert m.training_info["best_epoch"] <= 2
    assert m.training_info["stop_reason"] == "early_stopping"
    m = learner()
    m.fit(X, y)
    assert m.model.n_cv == 1 and m.model.val_fraction == 0 and m.model.n_refit == 0
    assert m.training_info["best_epoch"] is None
    assert np.isfinite(m.predict(X)).all()


@backend
def test_real_training_timeout():
    X, y = make_classification(n_samples=5000, n_features=20, random_state=42)
    m = learner()
    m.params["n_epochs"] = 10000
    m.fit(X[:4000], y[:4000], X_validation=X[4000:], y_validation=y[4000:], max_time=2)
    assert m.training_info["completed_epochs"] < 10000
    assert m.training_info["stop_reason"] == "time_limit"
    assert m.training_info["elapsed_seconds"] < 15
    assert np.isfinite(m.predict(X[4000:])).all()


@backend
def test_gpu_memory_rejection_before_training():
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    m = learner()
    m.params["device"] = "cuda:0"
    with patch("torch.cuda.is_available", return_value=True), patch(
        "torch.cuda.device_count", return_value=1
    ), patch("torch.cuda.mem_get_info", return_value=(1024, 8 * 1024**3)), patch(
        "supervised.algorithms._realmlp_backend.RealMLPClassifier.fit"
    ) as fit:
        with pytest.raises(RealMLPResourceError, match="GPU memory"):
            m.fit(X, y)
        fit.assert_not_called()


def test_cpu_threads_respect_effective_cpu_count():
    with patch("supervised.algorithms.realmlp_resources.cpu_count", return_value=4):
        assert thread_count(-1) == 4
        assert thread_count(-2) == 3
        assert thread_count(36) == 4
        assert thread_count(2) == 2
    with pytest.raises(ValueError):
        thread_count(0)


@backend
@pytest.mark.parametrize("generator", ["classification", "regression", "friedman"])
def test_learns_signal(generator):
    if generator == "classification":
        X, y = make_classification(
            n_samples=1000,
            n_features=10,
            n_informative=6,
            class_sep=2,
            flip_y=0,
            random_state=42,
        )
        m = learner()
    elif generator == "regression":
        X, y = make_regression(
            n_samples=1000, n_features=10, n_informative=6, noise=2, random_state=42
        )
        m = learner("regression")
    else:
        X, y = make_friedman1(n_samples=1000, n_features=10, noise=0.1, random_state=42)
        m = learner("regression")
    m.params["n_epochs"] = 64
    Xtr, Xtest, ytr, ytest = train_test_split(X, y, test_size=0.2, random_state=43)
    Xtr, Xval, ytr, yval = train_test_split(Xtr, ytr, test_size=0.25, random_state=44)
    m.fit(Xtr, ytr, X_validation=Xval, y_validation=yval)
    p = m.predict(Xtest)
    if generator == "classification":
        assert log_loss(ytest, p) < 0.5
    else:
        assert mean_squared_error(ytest, p) < 0.6 * mean_squared_error(
            ytest, np.full(len(ytest), np.mean(ytr))
        )


@backend
def test_gpu_portable_reload(tmp_path):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    X, y = make_classification(n_samples=200, n_features=10, random_state=42)
    m = learner()
    m.params["device"] = "cuda:0"
    m.fit(X[:150], y[:150], X_validation=X[150:], y_validation=y[150:])
    p = m.predict(X[150:])
    m.save(tmp_path / "gpu-model")
    m.load(tmp_path / "gpu-model")
    assert m.device == "cpu"
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always")
        assert_allclose(p, m.predict(X[150:]), atol=1e-5, rtol=1e-4)
    assert not any("GPU available but not used" in str(w.message) for w in emitted)
