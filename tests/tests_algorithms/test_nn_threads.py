import numpy as np
import pytest
from threadpoolctl import threadpool_info, threadpool_limits

from supervised.algorithms.nn import MLPAlgorithm, MLPRegressorAlgorithm


def blas_threads():
    pools = [p["num_threads"] for p in threadpool_info() if p["user_api"] == "blas"]
    assert pools
    return pools


@pytest.mark.parametrize(
    "cls,task",
    [
        (MLPAlgorithm, "binary_classification"),
        (MLPAlgorithm, "multiclass_classification"),
        (MLPRegressorAlgorithm, "regression"),
    ],
)
@pytest.mark.parametrize("jobs,expected", [(1, 1), (2, 2), (-1, 4), (-2, 3), (-20, 1)])
def test_native_limit_inside_fit_predict_and_restore(
    monkeypatch, cls, task, jobs, expected
):
    monkeypatch.setattr("supervised.algorithms.nn.cpu_count", lambda: 4)
    learner = cls({"ml_task": task, "n_jobs": jobs})
    seen = []

    def fit(X, y):
        seen.append(blas_threads())
        learner.model.n_iter_ = 1

    def predict(X):
        seen.append(blas_threads())
        return (
            np.zeros(len(X))
            if task == "regression"
            else np.ones((len(X), 3 if task == "multiclass_classification" else 2))
            / (3 if task == "multiclass_classification" else 2)
        )

    monkeypatch.setattr(learner.model, "fit", fit)
    monkeypatch.setattr(
        learner.model, "predict" if task == "regression" else "predict_proba", predict
    )
    with threadpool_limits(limits=4, user_api="blas"):
        before = blas_threads()
        learner.fit(np.zeros((12, 3)), np.arange(12) % 2)
        assert blas_threads() == before
        learner.predict(np.zeros((12, 3)))
        assert blas_threads() == before
    assert len(seen) == 2
    assert all(n == expected for pools in seen for n in pools)


@pytest.mark.parametrize("operation", ["fit", "predict"])
def test_limits_restore_on_exception(monkeypatch, operation):
    learner = MLPRegressorAlgorithm({"ml_task": "regression", "n_jobs": 1})
    learner.model.n_iter_ = 1

    def fail(*args):
        assert all(n == 1 for n in blas_threads())
        raise RuntimeError("injected failure")

    monkeypatch.setattr(learner.model, operation, fail)
    with threadpool_limits(limits=3, user_api="blas"):
        before = blas_threads()
        with pytest.raises(RuntimeError, match="injected failure"):
            if operation == "fit":
                learner.fit(np.zeros((12, 3)), np.zeros(12))
            else:
                learner.predict(np.zeros((12, 3)))
        assert blas_threads() == before


def test_optuna_passes_thread_limit(monkeypatch):
    import optuna
    from supervised.tuner.optuna.nn import NeuralNetworkObjective
    from supervised.utils.metric import Metric

    seen = []

    class Learner:
        def __init__(self, params):
            seen.append(params["n_jobs"])

        def fit(self, *args, **kwargs):
            pass

        def predict(self, X):
            return np.zeros(len(X))

    monkeypatch.setattr("supervised.tuner.optuna.nn.MLPRegressorAlgorithm", Learner)
    objective = NeuralNetworkObjective(
        "regression",
        np.zeros((20, 3)),
        np.zeros(20),
        None,
        np.zeros((10, 3)),
        np.ones(10),
        None,
        Metric({"name": "rmse"}),
        2,
        123,
    )
    value = objective(
        optuna.trial.FixedTrial(
            {
                "dense_1_size": 8,
                "dense_2_size": 4,
                "learning_rate": 0.01,
                "learning_rate_type": "constant",
                "alpha": 0.0001,
            }
        )
    )
    assert value == 1 and seen == [2]


@pytest.mark.parametrize(
    "task", ["regression", "binary_classification", "multiclass_classification"]
)
@pytest.mark.parametrize("mixed", [False, True])
def test_automl_explicit_nn_trains_and_reloads(tmp_path, task, mixed):
    from supervised import AutoML
    from tests.tests_automl.test_training_store_lifecycle import automl

    rng = np.random.default_rng(123)
    X = rng.normal(size=(150, 4))
    y = X[:, 0] + X[:, 1]
    if task != "regression":
        y = np.digitize(
            y,
            np.quantile(y, [0.5] if task == "binary_classification" else [0.33, 0.66]),
        )
    algorithms = ["Baseline", "Neural Network"] if mixed else ["Neural Network"]
    model = automl(tmp_path / "run", algorithms=algorithms, ml_task=task, n_jobs=2)
    model.fit(X, y)
    assert model._algorithms == algorithms
    assert any(
        m.learner_params["model_type"] == "Neural Network" for m in model._models
    )
    assert all(m.learner_params["n_jobs"] == 2 for m in model._models)
    np.testing.assert_allclose(
        model.predict(X),
        AutoML(results_path=str(tmp_path / "run")).predict(X),
        rtol=0,
        atol=0,
    )
    assert model.n_jobs == 2
