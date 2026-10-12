from unittest.mock import Mock

import pytest

from supervised.tuner.mljar_tuner import MljarTuner


def model(name, score=0.2, train_time=10):
    result = Mock()
    result.get_name.return_value = name
    result.get_type.return_value = "Nearest Neighbors"
    result.get_final_loss.return_value = score
    result.get_train_time.return_value = train_time
    result.get_worst_fairness.return_value = 0.8
    result.get_best_fairness.return_value = 0.9
    return result


@pytest.mark.parametrize("fairness_metric", [None, "demographic_parity_ratio"])
@pytest.mark.parametrize("scenario", ["no_models", "too_slow", "only_golden"])
def test_empty_selection_preserves_columns(fairness_metric, scenario):
    tuner = MljarTuner.__new__(MljarTuner)
    tuner._fairness_metric = fairness_metric
    models = {
        "no_models": [],
        "too_slow": [model("1_NearestNeighbors")],
        "only_golden": [model("1_NearestNeighbors_GoldenFeatures", train_time=0.5)],
    }[scenario]

    frame, algorithms = tuner.df_models_algorithms(
        models, time_limit=1, exclude_golden=True
    )

    assert frame.empty
    assert {"model", "score", "model_type", "name", "train_time"} <= set(frame.columns)
    assert len(algorithms) == 0
    if fairness_metric is not None:
        assert {"worst_fairness", "best_fairness"} <= set(frame.columns)


def test_golden_filter_keeps_and_sorts_original_models():
    tuner = MljarTuner.__new__(MljarTuner)
    tuner._fairness_metric = None
    models = [
        model("1_NearestNeighbors", score=0.3),
        model("2_NearestNeighbors_GoldenFeatures", score=0.1),
        model("3_NearestNeighbors", score=0.2),
    ]

    frame, algorithms = tuner.df_models_algorithms(models, exclude_golden=True)

    assert frame["name"].tolist() == ["3_NearestNeighbors", "1_NearestNeighbors"]
    assert algorithms.tolist() == ["Nearest Neighbors"]


def test_kmeans_features_skip_when_no_model_meets_time_limit(tmp_path):
    tuner = MljarTuner.__new__(MljarTuner)
    tuner._fairness_metric = None

    assert (
        tuner.get_kmeans_features_params(
            [model("1_NearestNeighbors", train_time=10)],
            str(tmp_path),
            total_time_limit=1,
        )
        == []
    )


def test_kmeans_features_not_entered_without_models(tmp_path, monkeypatch):
    tuner = MljarTuner.__new__(MljarTuner)
    generate_kmeans = Mock(side_effect=AssertionError("No models to improve"))
    monkeypatch.setattr(tuner, "get_kmeans_features_params", generate_kmeans)

    assert tuner.generate_params("kmeans_features", [], str(tmp_path), [], 60) == []
    generate_kmeans.assert_not_called()
