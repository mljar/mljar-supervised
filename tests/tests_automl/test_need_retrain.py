import json
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from supervised import AutoML
from supervised.algorithms.registry import (
    BINARY_CLASSIFICATION,
    MULTICLASS_CLASSIFICATION,
    REGRESSION,
)
from supervised.model_framework import ModelFramework
from supervised.utils.metric import Metric


@pytest.mark.parametrize("saved_loss", ["0.12345678901234567", 0.125, None])
def test_load_final_loss(tmp_path, saved_loss):
    model_path = tmp_path / "model"
    model_path.mkdir()
    desc = {
        "params": {"learner": {}, "validation_strategy": {}},
        "final_loss": saved_loss,
        "learners": [],
        "saved": [],
        "preprocessing": [],
    }
    (model_path / "framework.json").write_text(json.dumps(desc))

    model = ModelFramework.load(str(tmp_path), "model")

    if saved_loss is None:
        assert model.final_loss is None
    else:
        assert isinstance(model.get_final_loss(), float)
        assert model.get_final_loss() == float(saved_loss)


@pytest.mark.parametrize(
    "new_score, expected", [(0.9, False), (1.05, False), (1.2, True)]
)
def test_need_retrain_uses_binary_probabilities(monkeypatch, new_score, expected):
    model = AutoML(verbose=0)
    model._ml_task = BINARY_CLASSIFICATION
    X = pd.DataFrame({"feature": [0, 1]})
    y = pd.Series([0, 1])
    weights = pd.Series([1.0, 2.0])
    probabilities = np.array([[0.8, 0.2], [0.3, 0.7]])
    metric = Mock(return_value=new_score)
    model._best_model = Mock()
    model._best_model.get_metric.return_value = metric
    model._best_model.get_final_loss.return_value = 1.0
    predict_proba = Mock(return_value=probabilities)
    predict = Mock(side_effect=AssertionError("Expected probabilities"))
    monkeypatch.setattr(model, "_predict_proba", predict_proba)
    monkeypatch.setattr(model, "_predict", predict)

    assert bool(model.need_retrain(X, y, sample_weight=weights)) == expected

    predict_proba.assert_called_once_with(X)
    predict.assert_not_called()
    actual_y, actual_predictions, actual_weights = metric.call_args.args
    assert actual_y is y
    assert actual_weights is weights
    np.testing.assert_array_equal(actual_predictions, probabilities[:, 1])


@pytest.mark.parametrize(
    "task, metric_name",
    [
        (REGRESSION, "rmse"),
        (BINARY_CLASSIFICATION, "logloss"),
        (MULTICLASS_CLASSIFICATION, "logloss"),
    ],
)
def test_need_retrain_after_reload(tmp_path, task, metric_name):
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(60, 3)), columns=["a", "b", "c"])
    if task == REGRESSION:
        y = pd.Series(X.a + rng.normal(size=60))
    else:
        y = pd.Series(np.arange(60) % (2 if task == BINARY_CLASSIFICATION else 3))
    results_path = str(tmp_path / "automl")
    model = AutoML(
        results_path=results_path,
        ml_task=task,
        eval_metric=metric_name,
        algorithms=["Baseline"],
        train_ensemble=False,
        explain_level=0,
        features_selection=False,
        golden_features=False,
        hill_climbing_steps=0,
        start_random_models=1,
        validation_strategy={"validation_type": "split", "train_ratio": 0.8},
        verbose=0,
        n_jobs=1,
    )
    model.fit(X, y)
    predictions = model.predict(X) if task == REGRESSION else model.predict_proba(X)
    if task == BINARY_CLASSIFICATION:
        predictions = predictions[:, 1]
    new_score = Metric({"name": metric_name})(y, predictions)
    old_score = model._best_model.get_final_loss()
    expected = new_score > old_score and abs((old_score - new_score) / old_score) > 0.1

    # Existing framework files store the validation loss as a JSON string.
    framework_path = (
        tmp_path / "automl" / model._best_model.get_name() / "framework.json"
    )
    desc = json.loads(framework_path.read_text())
    desc["final_loss"] = str(old_score)
    framework_path.write_text(json.dumps(desc))

    reloaded = AutoML(results_path=results_path, verbose=0)
    # need_retrain should load saved models without a preceding predict call.
    assert bool(reloaded.need_retrain(X, y)) == bool(expected)
    assert isinstance(reloaded._best_model.get_final_loss(), float)
    assert bool(model.need_retrain(X, y)) == bool(expected)
