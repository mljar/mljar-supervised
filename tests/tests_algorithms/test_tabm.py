import importlib.util
import json

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from scipy import sparse
from sklearn.datasets import make_classification, make_regression

from supervised.algorithms.tabm import (
    TabMClassificationAlgorithm,
    TabMRegressionAlgorithm,
    validate_options,
)
from supervised.algorithms.tabm_resources import TabMResourceError

backend = pytest.mark.skipif(
    importlib.util.find_spec("pytabkit") is None, reason="optional backend"
)


def learner(task="binary_classification", **params):
    cls = (
        TabMRegressionAlgorithm if task == "regression" else TabMClassificationAlgorithm
    )
    return cls(
        dict(
            ml_task=task,
            n_epochs=3,
            n_jobs=2,
            seed=42,
            **{"device": "cpu", "d_block": 64, **params}
        )
    )


@pytest.mark.parametrize(
    "params",
    [
        {"arch_type": "tabm"},
        {"n_blocks": 1},
        {"d_block": 512},
        {"tabm_k": 0},
        {"batch_size": 0},
        {"patience": -1},
        {"lr": 0},
        {"dropout": 1},
        {"memory_fraction": 0},
        {"device": "mps"},
    ],
)
def test_invalid_options(params):
    with pytest.raises(ValueError):
        validate_options(params)


@backend
@pytest.mark.parametrize(
    "task", ["binary_classification", "multiclass_classification", "regression"]
)
def test_training_reload_copy_and_history(task, tmp_path):
    if task == "regression":
        X, y = make_regression(n_samples=200, n_features=6, random_state=42)
    else:
        X, y = make_classification(
            n_samples=200,
            n_features=6,
            n_informative=4,
            n_redundant=0,
            n_classes=3 if task.startswith("multi") else 2,
            random_state=42,
        )
    X = pd.DataFrame(X)
    X["category"] = pd.Series(np.where(X[0] > 0, "a", "b"), dtype="category")
    X.loc[::8, 1] = np.nan
    X.loc[190:, "category"] = np.nan
    m = learner(task)
    log = str(tmp_path / "training.log")
    m.fit(
        X[:160],
        y[:160],
        X_validation=X[160:],
        y_validation=y[160:],
        log_to_file=log,
        max_time=30,
    )
    predictions = m.predict(X[160:])
    assert np.isfinite(predictions).all()
    assert predictions.shape == ((40, 3) if task.startswith("multi") else (40,))
    info = json.load(open(log + ".json"))
    if task != "regression":
        from sklearn.metrics import log_loss
        best_score = info["history"][info["best_epoch"] - 1]["validation"]
        assert_allclose(best_score, log_loss(y[160:], predictions), rtol=1e-5)
    assert info["arch_type"] == "tabm-mini"
    assert info["completed_epochs"] == 3
    assert 1 <= info["best_epoch"] <= 3
    assert_allclose(predictions, m.copy().predict(X[160:]), atol=1e-6)
    m.save(str(tmp_path / "model.tabm"))
    loaded = learner(task)
    loaded.load(str(tmp_path / "model.tabm"))
    assert_allclose(predictions, loaded.predict(X[160:]), atol=1e-6)


@backend
def test_sparse_weights_and_missing_validation_rejected():
    X, y = make_classification(n_samples=100, n_features=6, random_state=42)
    for kw in [
        {"sample_weight": np.ones(80)},
        {"sample_weight_validation": np.ones(20)},
    ]:
        with pytest.raises(ValueError, match="weights"):
            learner().fit(
                X[:80], y[:80], X_validation=X[80:], y_validation=y[80:], **kw
            )
    with pytest.raises(ValueError, match="validation"):
        learner().fit(X, y)
    with pytest.raises(ValueError, match="sparse"):
        learner().fit(
            sparse.csr_matrix(X[:80]), y[:80], X_validation=X[80:], y_validation=y[80:]
        )
    with pytest.raises(TabMResourceError):
        learner(memory_limit_gb=0.001).fit(
            X[:80], y[:80], X_validation=X[80:], y_validation=y[80:]
        )


@backend
def test_early_stop_restores_best_checkpoint(monkeypatch):
    from supervised.algorithms._tabm_backend import Metrics

    X, y = make_classification(n_samples=120, n_features=6, random_state=42)
    m = learner(patience=0)
    import torch

    monkeypatch.setattr(Metrics, "apply", lambda *args: torch.tensor(1.0))
    m.fit(X[:90], y[:90], X_validation=X[90:], y_validation=y[90:])
    assert m.training_info["best_epoch"] == 1
    assert m.training_info["completed_epochs"] == 2
    assert m.training_info["stop_reason"] == "early_stopping"
    # With an identical seed, the restored checkpoint must equal a one-epoch fit.
    first = learner()
    first.params["n_epochs"] = 1
    first.fit(X[:90], y[:90], X_validation=X[90:], y_validation=y[90:])
    assert_allclose(m.predict(X), first.predict(X), atol=1e-6)


@backend
def test_time_limit_restores_checkpoint(monkeypatch):
    from supervised.algorithms import _tabm_backend as backend_module

    X, y = make_classification(n_samples=120, n_features=6, random_state=42)
    original = backend_module.Metrics.apply
    original_time = backend_module.time.monotonic

    def score(*args):
        value = original(*args)
        # Expire the budget immediately after the first validation completes.
        monkeypatch.setattr(
            backend_module.time, "monotonic", lambda: original_time() + 1000
        )
        return value

    monkeypatch.setattr(backend_module.Metrics, "apply", score)
    m = learner()
    m.fit(X[:90], y[:90], X_validation=X[90:], y_validation=y[90:], max_time=60)
    assert m.training_info["completed_epochs"] == 1
    assert m.training_info["stop_reason"] == "time_limit"
    assert np.isfinite(m.predict(X)).all()


@backend
def test_cuda_reload_on_cpu(tmp_path):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    X, y = make_classification(n_samples=120, n_features=6, random_state=42)
    m = learner(device="cuda", predict_batch_size=17)
    m.fit(X[:90], y[:90], X_validation=X[90:], y_validation=y[90:])
    before = m.predict(X)
    m.save(str(tmp_path / "gpu.tabm"))
    restored = learner()
    restored.load(str(tmp_path / "gpu.tabm"))
    assert restored.device == "cpu"
    assert_allclose(before, restored.predict(X), atol=1e-5)


@backend
@pytest.mark.parametrize("kind", ["categorical", "constant_numeric"])
def test_degenerate_columns_and_unknown_categories(kind):
    X = pd.DataFrame({"constant": np.ones(100), "category": ["a", "b"] * 50})
    if kind == "categorical":
        X = X[["category"]]
    y = np.tile([0, 1], 50)
    m = learner(tabm_k=32, memory_limit_gb=None)
    m.fit(X[:80], y[:80], X_validation=X[80:], y_validation=y[80:])
    unseen = X[80:].copy()
    unseen["category"] = "unseen"
    assert np.isfinite(m.predict(unseen)).all()


@backend
def test_exhausted_time_budget_has_no_model():
    from supervised.exceptions import BudgetExceededError

    X, y = make_classification(n_samples=100, n_features=6, random_state=42)
    m = learner()
    with pytest.raises(BudgetExceededError, match="time budget"):
        m.fit(X[:80], y[:80], X_validation=X[80:], y_validation=y[80:], max_time=0)
    assert not m.is_fitted()


@backend
def test_estimate_grows_with_network_and_ensemble():
    from supervised.algorithms._tabm_backend import TabMClassifier
    from supervised.algorithms.tabm_resources import check_resources

    X = pd.DataFrame(np.ones((100, 10)))
    small = TabMClassifier(n_blocks=2, d_block=64, tabm_k=16, batch_size="auto")
    large = TabMClassifier(n_blocks=3, d_block=128, tabm_k=32, batch_size="auto")
    a = check_resources(small, X, X, 2, "cpu", {})
    b = check_resources(large, X, X, 2, "cpu", {})
    assert a["estimated_ram_bytes"] < b["estimated_ram_bytes"]
