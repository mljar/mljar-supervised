import warnings
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from supervised.algorithms.knn import (
    KNeighborsAlgorithm,
    KNeighborsRegressorAlgorithm,
)
from supervised.algorithms.random_forest import RandomForestAlgorithm
from supervised.utils.importance import PermutationImportance
from supervised.utils.shap import PlotSHAP


@pytest.fixture(
    params=[
        (KNeighborsAlgorithm, "binary_classification"),
        (KNeighborsAlgorithm, "multiclass_classification"),
        (KNeighborsRegressorAlgorithm, "regression"),
    ]
)
def model(request):
    algorithm, task = request.param
    return algorithm({"ml_task": task, "n_jobs": 1})


@pytest.mark.parametrize(
    "train_rows,validation_rows,skip",
    [
        (100, 100, False),
        (10000, 100, False),
        (100, 10000, False),
        (10001, 100, True),
        (100, 10001, True),
        (10001, 10001, True),
    ],
)
@pytest.mark.parametrize("explain_level", [0, 1, 2])
def test_interpretation_row_limit(
    model, train_rows, validation_rows, skip, explain_level
):
    X_train = pd.DataFrame(np.zeros((train_rows, 2)))
    y_train = pd.Series(np.arange(train_rows) % 2)
    X_validation = pd.DataFrame(np.zeros((validation_rows, 2)))
    y_validation = pd.Series(np.arange(validation_rows) % 2)
    with patch.object(
        PermutationImportance, "compute_and_plot"
    ) as importance, patch.object(PlotSHAP, "compute") as shap:
        with warnings.catch_warnings(record=True) as emitted:
            warnings.simplefilter("always")
            model.interpret(
                X_train,
                y_train,
                X_validation,
                y_validation,
                "results",
                "learner",
                metric_name="metric",
                ml_task=model.ml_task,
                explain_level=explain_level,
            )
        shap.assert_not_called()
        if skip or explain_level == 0:
            importance.assert_not_called()
        else:
            importance.assert_called_once_with(
                model,
                X_validation,
                y_validation,
                "results",
                "learner",
                "metric",
                model.ml_task,
                1,
            )
        if skip and explain_level > 0:
            assert len(emitted) == 1
            message = str(emitted[0].message)
            assert f"training rows={train_rows}" in message
            assert f"validation rows={validation_rows}" in message
            assert "row limit=10000" in message
        else:
            assert emitted == []


def test_knn_shap_is_unavailable_without_attempting_explainer(model):
    X = pd.DataFrame(np.zeros((100, 2)))
    y = pd.Series(np.arange(100) % 2)
    # Exercise the algorithm guard even when SHAP is installed and available.
    with patch("supervised.utils.shap.shap_pacakge_available", True), patch.object(
        PlotSHAP, "get_explainer"
    ) as explainer:
        assert not PlotSHAP.is_available(model, X, y, model.ml_task)
        PlotSHAP.compute(model, X, y, X, y, "results", "learner", None, model.ml_task)
        explainer.assert_not_called()


def test_other_algorithms_still_request_interpretation_on_large_data():
    model = RandomForestAlgorithm({"ml_task": "binary_classification", "n_jobs": 1})
    X = pd.DataFrame(np.zeros((10001, 2)))
    y = pd.Series(np.arange(10001) % 2)
    with patch.object(
        PermutationImportance, "compute_and_plot"
    ) as importance, patch.object(PlotSHAP, "compute") as shap:
        model.interpret(X, y, X, y, "results", "learner", explain_level=2)
        importance.assert_called_once()
        shap.assert_called_once()


def test_training_subsample_does_not_bypass_interpretation_limit():
    model = KNeighborsAlgorithm({"ml_task": "binary_classification", "n_jobs": 1})
    X = pd.DataFrame(np.arange(10001, dtype=float).reshape(-1, 1))
    y = pd.Series(np.arange(10001) % 2)
    model.fit(X, y)
    assert model.model.n_samples_fit_ == 1000
    before = model.predict(X.iloc[:10])
    with patch.object(
        PermutationImportance, "compute_and_plot"
    ) as importance, patch.object(PlotSHAP, "compute") as shap:
        with pytest.warns(UserWarning, match="training rows=10001"):
            model.interpret(X, y, X.iloc[:10], y.iloc[:10], "results", "learner")
        importance.assert_not_called()
        shap.assert_not_called()
    np.testing.assert_array_equal(before, model.predict(X.iloc[:10]))
