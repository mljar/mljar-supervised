from unittest.mock import Mock

import numpy as np
import pytest
from catboost.utils import eval_metric as catboost_metric
from sklearn.metrics import f1_score

from supervised.algorithms.catboost import catboost_eval_metric
from supervised.utils.metric import (
    F1_AVERAGES,
    F1_METRICS,
    Metric,
    lightgbm_eval_metric_f1,
    xgboost_eval_metric_f1,
)


@pytest.mark.parametrize("name", F1_METRICS)
@pytest.mark.parametrize("classes", [2, 3])
@pytest.mark.parametrize("weighted", [False, True])
def test_f1_matches_sklearn_and_callbacks(name, classes, weighted):
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1] + ([2, 2] if classes == 3 else []))
    predicted = np.array([0, 0, 0, 0, 1, 1, 0, 1] + ([1, 2] if classes == 3 else []))
    probabilities = np.full((len(y), classes), 0.1 / (classes - 1))
    probabilities[np.arange(len(y)), predicted] = 0.9
    weights = np.arange(1, len(y) + 1, dtype=float) if weighted else None
    average = F1_AVERAGES.get(name, "micro" if classes == 3 else "binary")
    expected = f1_score(
        y, predicted, average=average, sample_weight=weights, zero_division=0
    )
    inputs = probabilities if classes == 3 else probabilities[:, 1]
    metric = Metric({"name": name})
    assert metric(y, inputs, weights) == pytest.approx(-expected)
    assert metric.get_minimize_direction()
    assert metric.is_negative()
    assert Metric.optimize_negative(name)
    assert metric.improvement(-0.1, -0.2)
    data = Mock()
    data.get_label.return_value = y
    data.get_weight.return_value = np.array([]) if weights is None else weights
    assert xgboost_eval_metric_f1(inputs, data, metric_name=name) == (
        name,
        pytest.approx(-expected),
    )
    data.get_weight.return_value = weights
    for lgb_inputs in (
        [inputs, probabilities.ravel(order="F")] if classes == 3 else [inputs]
    ):
        assert lightgbm_eval_metric_f1(lgb_inputs, data, metric_name=name) == (
            name,
            pytest.approx(expected),
            True,
        )
    raw = np.log(probabilities) if classes == 3 else np.log(inputs / (1 - inputs))
    native = catboost_metric(
        y,
        raw,
        catboost_eval_metric(
            "multiclass_classification" if classes == 3 else "binary_classification",
            name,
        ),
        weight=weights,
    )[0]
    assert native == pytest.approx(expected)


@pytest.mark.parametrize("name", F1_AVERAGES)
def test_missing_fold_class_is_included(name):
    y = np.array([0, 0, 1, 1])
    probabilities = np.array(
        [[0.8, 0.1, 0.1], [0.6, 0.3, 0.1], [0.7, 0.2, 0.1], [0.1, 0.8, 0.1]]
    )
    expected = f1_score(
        y,
        probabilities.argmax(axis=1),
        average=F1_AVERAGES[name],
        labels=[0, 1, 2],
        zero_division=0,
    )
    assert -Metric({"name": name})(y, probabilities) == pytest.approx(expected)
    assert catboost_metric(
        y,
        np.log(probabilities),
        catboost_eval_metric("multiclass_classification", name),
    )[0] == pytest.approx(expected)
    data = Mock()
    data.get_label.return_value = y
    data.get_weight.return_value = None
    for predictions in [probabilities, probabilities.ravel(order="F")]:
        assert lightgbm_eval_metric_f1(predictions, data, metric_name=name)[
            1
        ] == pytest.approx(expected)


def test_macro_f1_can_prefer_a_different_model_from_accuracy():
    y = np.array([0] * 8 + [1, 2])
    majority = np.eye(3)[np.zeros(10, dtype=int)]
    minority = np.eye(3)[[0] * 5 + [1] * 3 + [1, 2]]
    accuracy = Metric({"name": "accuracy"})
    macro = Metric({"name": "f1_macro"})
    assert accuracy(y, majority) < accuracy(y, minority)
    assert macro(y, minority) < macro(y, majority)
    assert Metric({"name": "f1"})(y, majority) == accuracy(y, majority)


@pytest.mark.parametrize("name", F1_AVERAGES)
def test_f1_variants_rejected_for_regression(name):
    from supervised import AutoML

    model = AutoML(ml_task="regression", eval_metric=name, verbose=0)
    with pytest.raises(ValueError, match="not allowed"):
        model._validate_eval_metric()
