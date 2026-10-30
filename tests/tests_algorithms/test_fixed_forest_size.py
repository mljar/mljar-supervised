import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from sklearn.datasets import make_classification, make_regression

from supervised.algorithms.extra_trees import ExtraTreesAlgorithm, ExtraTreesRegressorAlgorithm
from supervised.algorithms.random_forest import RandomForestAlgorithm, RandomForestRegressorAlgorithm
from supervised.algorithms.sklearn import SklearnTreesEnsembleClassifierAlgorithm


@pytest.mark.parametrize("rows,expected", [
    (1, 100), (999, 100), (1000, 100), (1999, 100),
    (2000, 200), (2999, 200), (3000, 300), (10_000_000, 300),
])
def test_tree_count_boundaries(rows, expected):
    assert SklearnTreesEnsembleClassifierAlgorithm.tree_count(rows) == expected


@pytest.mark.parametrize("algorithm,task", [
    (RandomForestAlgorithm, "binary_classification"),
    (ExtraTreesAlgorithm, "binary_classification"),
    (RandomForestAlgorithm, "multiclass_classification"),
    (ExtraTreesAlgorithm, "multiclass_classification"),
    (RandomForestRegressorAlgorithm, "regression"),
    (ExtraTreesRegressorAlgorithm, "regression"),
])
def test_fixed_size_uses_complete_split_and_ignores_validation_labels(algorithm, task, tmp_path):
    if task == "regression":
        X, y = make_regression(n_samples=2000, n_features=4, random_state=7)
        metric = "rmse"
    else:
        X, y = make_classification(n_samples=2000, n_features=4, n_informative=3,
                                   n_redundant=0, n_classes=3 if task.startswith("multi") else 2,
                                   random_state=7)
        metric = "logloss"
    params = dict(ml_task=task, eval_metric_name=metric, max_depth=3, n_jobs=1, seed=7)
    weights = np.linspace(0.5, 1.5, 1800)
    val_weights = np.linspace(0.5, 1.5, 200)
    model = algorithm(params.copy())
    log = tmp_path / "training.log"
    model.fit(X[:1800], y[:1800], sample_weight=weights,
              X_validation=X[1800:], y_validation=y[1800:],
              sample_weight_validation=val_weights, log_to_file=str(log))
    # Counting only the 1800 training rows would incorrectly select 100 trees.
    assert len(model.model.estimators_) == model.model.n_estimators == 200
    assert model.params["n_estimators"] == 200
    values = pd.read_csv(log, header=None)
    assert len(values) == 1 and values.iloc[0, 0] == 199
    sign = -1 if model.log_metric.is_negative() else 1
    assert_allclose(values.iloc[0, 2], sign * model.log_metric(
        y[1800:], model.predict(X[1800:]), sample_weight=val_weights))

    changed = algorithm(params.copy())
    changed.fit(X[:1800], y[:1800], sample_weight=weights,
                X_validation=X[1800:], y_validation=y[1800:][::-1])
    assert_allclose(model.predict(X), changed.predict(X), rtol=0, atol=0)
    path = tmp_path / model.get_fname()
    model.save(str(path))
    loaded = algorithm(params.copy())
    loaded.load(str(path))
    assert len(loaded.model.estimators_) == 200
    assert_allclose(model.predict(X), loaded.predict(X), rtol=0, atol=0)


@pytest.mark.parametrize("algorithm", [RandomForestAlgorithm, ExtraTreesAlgorithm])
def test_fit_without_validation_and_refit_does_not_accumulate_trees(algorithm):
    X, y = make_classification(n_samples=3000, n_features=4, random_state=7)
    model = algorithm(dict(ml_task="binary_classification", max_depth=2, n_jobs=1, seed=7))
    model.fit(X, y)
    assert len(model.model.estimators_) == 300
    model.fit(X[:1000], y[:1000])
    assert len(model.model.estimators_) == model.model.n_estimators == 100
