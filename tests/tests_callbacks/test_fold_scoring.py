from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from supervised.callbacks.early_stopping import EarlyStopping
from supervised.ensemble import Ensemble
from supervised.model_framework import ModelFramework
from supervised.utils.metric import Metric
from supervised.utils.validation_scores import collect_fold_predictions, mean_fold_score


def finish_folds(metric, folds):
    callback = EarlyStopping({"metric": {"name": metric}, "keep_best_model": False})
    for i, (index, target, prediction, weights) in enumerate(folds):
        callback.add_and_set_learner(SimpleNamespace(uid=str(i)))
        callback.on_learner_train_start(None)
        callback.on_iteration_end(
            {"iter_cnt": 0},
            {
                "y_validation_true": np.asarray(target),
                "y_validation_predicted": np.asarray(prediction),
                "sample_weight_validation": weights,
                "validation_index": index,
                "validation_columns": (
                    [f"prediction_{j}" for j in range(np.asarray(prediction).shape[1])]
                    if np.asarray(prediction).ndim == 2 else "prediction"
                ),
            },
        )
    callback.on_framework_train_end(None)
    return callback


def test_repeats_score_each_fold_equally_before_averaging_predictions():
    folds = [
        ([0, 1], [0, 0], [1, 1], None),
        ([2], [0], [3], None),
        ([0, 1], [0, 0], [-5, -5], None),
        ([2], [0], [-7], None),
    ]
    callback = finish_folds("rmse", folds)
    # Equal weight for all four folds, even though their sizes differ.
    assert callback.final_loss == pytest.approx((1 + 3 + 5 + 7) / 4)
    np.testing.assert_array_equal(callback.best_y_oof.prediction, [-2, -2, -2])
    assert len(callback.validation_predictions) == 6


@pytest.mark.parametrize(
    "metric,target,predictions",
    [
        ("rmse", [0, 1, 2], [[1, 2, 5], [2, 3, 1]]),
        ("auc", [0, 1, 0], [[0.1, 0.2, 0.3], [0.3, 0.9, 0.2]]),
        ("logloss", [0, 1, 2], [
            [[0.6, 0.3, 0.1], [0.2, 0.5, 0.3], [0.1, 0.1, 0.8]],
            [[0.2, 0.5, 0.3], [0.1, 0.8, 0.1], [0.4, 0.4, 0.2]],
        ]),
    ],
)
def test_weighted_fold_metrics_and_metric_direction(metric, target, predictions):
    weights = [[1, 2, 3], [3, 2, 1]]
    callback = finish_folds(metric, [
        (range(3), target, prediction, weight)
        for prediction, weight in zip(predictions, weights)
    ])
    expected = np.mean([
        Metric({"name": metric})(np.asarray(target), np.asarray(p), np.asarray(w))
        for p, w in zip(predictions, weights)
    ])
    assert callback.final_loss == pytest.approx(expected)
    if metric == "auc":
        assert callback.final_loss < 0  # Internal selection minimizes negative AUC.


def test_single_repeat_uses_mean_of_fold_scores_too():
    callback = finish_folds("auc", [
        ([0, 1], [0, 1], [0.1, 0.2], None),
        ([2, 3], [0, 1], [0.8, 0.9], None),
    ])
    assert callback.final_loss == -1
    assert Metric({"name": "auc"})(
        callback.best_y_oof.target, callback.best_y_oof.prediction
    ) == -0.75


@pytest.mark.parametrize("metric", ["rmse", "mse"])
def test_perfect_fold_has_zero_error_and_does_not_hide_other_folds(metric):
    callback = finish_folds(metric, [
        ([0, 1], [0, 1], [0, 1], None),
        ([2, 3], [0, 1], [2, 3], None),
    ])
    error = Metric({"name": metric})(np.array([0, 1]), np.array([2, 3]))
    assert callback.final_loss == pytest.approx(error / 2)
    assert callback.final_loss > 0


def test_ensemble_selects_on_fold_scores_not_cancellation_across_repeats(monkeypatch):
    unstable = collect_fold_predictions([
        pd.DataFrame({"target": [0, 0], "prediction": [4, 4]}),
        pd.DataFrame({"target": [0, 0], "prediction": [-4, -4]}),
    ])
    stable = unstable.copy()
    stable["prediction"] = 1.0
    models = [
        SimpleNamespace(
            get_name=lambda name=name: name,
            get_validation_predictions=lambda frame=frame: frame.copy(),
        )
        for name, frame in [("unstable", unstable), ("stable", stable)]
    ]
    ensemble = Ensemble("mse", "regression")
    monkeypatch.setattr(ensemble, "get_additional_metrics", lambda: {})
    oofs, target, weights, sensitive = ensemble.get_oof_matrix(models)
    ensemble.fit(oofs, target, weights, sensitive)
    assert ensemble.best_algs[0] == "stable"
    assert ensemble.best_algs.count("stable") / len(ensemble.best_algs) > 0.9
    assert ensemble.best_loss == pytest.approx(mean_fold_score(
        ensemble.metric, target, ensemble.total_best_sum
    ))
    assert ensemble.best_loss > 0.9
    assert len(ensemble.get_validation_predictions()) == 4
    assert len(ensemble.get_out_of_folds()) == 2


def test_ensemble_rejects_mismatched_validation_folds():
    frame = collect_fold_predictions([pd.DataFrame({"target": [0, 1], "prediction": [1, 2]})])
    different = frame.copy()
    different.index = pd.MultiIndex.from_tuples([(0, 2), (0, 3)], names=frame.index.names)
    models = [SimpleNamespace(
        get_name=lambda i=i: str(i), get_validation_predictions=lambda f=f: f,
    ) for i, f in enumerate([frame, different])]
    from supervised.exceptions import NotTrainedException

    with pytest.raises(NotTrainedException, match="same validation folds"):
        Ensemble("mse", "regression").get_oof_matrix(models)


def test_one_hot_targets_are_normalized_for_ensembling_without_changing_fold_rows():
    callback = EarlyStopping({"metric": {"name": "logloss"}})
    frame = pd.DataFrame({
        "target_0": [1, 0, 0], "target_1": [0, 1, 0], "target_2": [0, 0, 1],
        "prediction_0": [0.6, 0.2, 0.1], "prediction_1": [0.3, 0.5, 0.1],
        "prediction_2": [0.1, 0.3, 0.8],
    })
    callback.validation_predictions = collect_fold_predictions([frame, frame])
    model = ModelFramework({"learner": {}, "validation_strategy": {}}, [callback])
    predictions = model.get_validation_predictions()
    assert predictions.index.equals(callback.validation_predictions.index)
    np.testing.assert_array_equal(predictions.target, [0, 1, 2, 0, 1, 2])
    assert "target_0" in callback.validation_predictions


def test_ensemble_does_not_mix_old_and_new_score_definitions():
    from supervised.exceptions import NotTrainedException

    frame = pd.DataFrame({"target": [0, 1], "prediction": [1, 2]})
    old = SimpleNamespace(get_name=lambda: "old", get_out_of_folds=lambda: frame)
    new = SimpleNamespace(get_name=lambda: "new", get_validation_predictions=lambda: collect_fold_predictions([frame]))
    with pytest.raises(NotTrainedException, match="Cannot mix legacy"):
        Ensemble("mse", "regression").get_oof_matrix([old, new])
