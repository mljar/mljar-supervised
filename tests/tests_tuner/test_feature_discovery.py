from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from supervised import AutoML
from supervised.exceptions import AutoMLException
from supervised.tuner.mljar_tuner import MljarTuner
from supervised.tuner.time_controller import TimeController


def test_budget_is_shared_without_changing_other_steps():
    original = TimeController(
        0, 3600, None, ["default_algorithms", "golden_features", "stack"], ["LightGBM"]
    )
    combined = TimeController(
        0,
        3600,
        None,
        ["default_algorithms", "golden_features", "feature_discovery", "stack"],
        ["LightGBM"],
    )
    discovery = TimeController(
        0,
        3600,
        None,
        ["default_algorithms", "feature_discovery", "stack"],
        ["LightGBM"],
    )
    assert combined.time_should_use("golden_features") == pytest.approx(
        original.time_should_use("golden_features") * 3 / 8
    )
    assert combined.time_should_use("feature_discovery") == pytest.approx(
        original.time_should_use("golden_features") * 5 / 8
    )
    assert discovery.time_should_use("feature_discovery") == original.time_should_use(
        "golden_features"
    )
    assert combined.time_should_use("stack") == original.time_should_use("stack")
    restored = TimeController.from_json(combined.to_json())
    assert restored.time_should_use("feature_discovery") == combined.time_should_use(
        "feature_discovery"
    )


@pytest.mark.parametrize("golden", [False, True])
@pytest.mark.parametrize("discovery", [False, True])
@pytest.mark.parametrize("mix", [False, True])
@pytest.mark.parametrize("kmeans", [False, True])
def test_feature_budget_preserves_total_and_other_steps(golden, discovery, mix, kmeans):
    weights = {"default_algorithms": 0.3, "stack": 0.2}
    if golden:
        weights["golden_features"] = 0.025 if discovery else 0.05
    if discovery:
        weights["feature_discovery"] = 0.025 if golden else 0.05
    if mix:
        weights["mix_encoding"] = 0.05
    if kmeans:
        weights["kmeans_features"] = 0.05
    original_weights = weights.copy()
    if mix and discovery:
        weights["mix_encoding"] -= 0.025
        weights["feature_discovery"] += 0.025
    total_weight = sum(original_weights.values())
    controller = TimeController(0, 3600, None, list(weights), ["LightGBM"])
    restored = TimeController.from_json(controller.to_json())
    feature_shares = {"golden_features": 3, "feature_discovery": 5, "kmeans_features": 2}
    enabled = [step for step in feature_shares if step in weights]
    feature_time = sum(controller.time_should_use(step) for step in enabled)
    assert feature_time == pytest.approx(
        3600 * sum(weights[step] for step in enabled) / total_weight
    )
    for step in enabled:
        assert controller.time_should_use(step) == pytest.approx(
            feature_time * feature_shares[step] / sum(feature_shares[s] for s in enabled)
        )
    for step, weight in weights.items():
        assert restored.time_should_use(step) == pytest.approx(
            controller.time_should_use(step)
        )
        if step not in feature_shares:
            assert controller.time_should_use(step) == pytest.approx(
                3600 * weight / total_weight
            )
    assert sum(controller.time_should_use(s) for s in weights) == pytest.approx(3600)


def test_compete_discovery_budget_with_all_stages():
    steps = [
        "adjust_validation",
        "simple_algorithms",
        "default_algorithms",
        "not_so_random",
        "mix_encoding",
        "golden_features",
        "feature_discovery",
        "kmeans_features",
        "insert_random_feature",
        "features_selection",
        "hill_climbing_1",
        "hill_climbing_2",
        "boost_on_errors",
        "ensemble",
        "stack",
        "ensemble_stacked",
    ]
    controller = TimeController(0, 3600, None, steps, ["LightGBM", "Xgboost"])
    assert controller.time_should_use("mix_encoding") == pytest.approx(3600 / 54)
    assert controller.time_should_use("golden_features") == pytest.approx(100)
    assert controller.time_should_use("feature_discovery") == pytest.approx(500 / 3)
    assert controller.time_should_use("kmeans_features") == pytest.approx(200 / 3)
    assert sum(controller.time_should_use(step) for step in steps) == pytest.approx(
        3600
    )


def parent(name, score, golden=False, train_time=2):
    params = {
        "name": name,
        "learner": {
            "model_type": "LightGBM",
            "seed": 42,
            "learning_rate": 0.1,
            "max_depth": int(score),
        },
        "preprocessing": {"golden_features": {"features_count": 5}} if golden else {},
        "validation_strategy": {"validation_type": "split"},
        "optuna_time_budget": 10,
    }
    return SimpleNamespace(
        params=params,
        validation_params=params["validation_strategy"],
        _is_stacked=False,
        get_name=lambda: name,
        get_type=lambda: "LightGBM",
        get_final_loss=lambda: score,
        get_train_time=lambda: train_time,
    )


@pytest.mark.parametrize("elapsed, train_time", [(650, 60), (650, 40), (700, 2)])
def test_discovery_is_not_started_without_time_to_train(
    monkeypatch, elapsed, train_time
):
    model = AutoML(verbose=0)
    model._verbose = 0
    model._total_time_limit = 700
    model._model_time_limit = None
    model._models = [parent("1_LightGBM", 1, train_time=train_time)]
    model.tuner = object.__new__(MljarTuner)
    model.tuner._fairness_metric = None
    model.tuner._top_models_to_improve = 3
    model._time_ctrl = TimeController(
        0,
        700,
        None,
        ["default_algorithms", "not_so_random", "feature_discovery"],
        ["LightGBM"],
    )
    # At 650s, discovery has 50s left, including 12.5s for generation.
    monkeypatch.setattr("supervised.tuner.time_controller.time.time", lambda: elapsed)
    generator = Mock(side_effect=AssertionError("No affordable discovery parent"))
    monkeypatch.setattr("supervised.base_automl.FeatureDiscovery", generator)
    assert model._prepare_feature_discovery() == []
    generator.assert_not_called()


def test_discovery_selects_affordable_runners_up_before_top_model_limit():
    tuner = object.__new__(MljarTuner)
    tuner._fairness_metric = None
    tuner._top_models_to_improve = 1
    models = [
        parent("1_LightGBM", 1, train_time=200),
        parent("2_LightGBM", 2, train_time=100),
        parent("3_LightGBM", 3, train_time=60),
        parent("4_LightGBM_GoldenFeatures", 4, True, train_time=10),
        parent("5_LightGBM", 5, train_time=5),
    ]
    chosen = tuner.feature_discovery_parents(models, max_train_time=10)
    assert [m.get_name() for m in chosen] == ["4_LightGBM_GoldenFeatures"]
    assert "golden_features" in chosen[0].params["preprocessing"]
    tuner._top_models_to_improve = 3
    assert [m.get_name() for m in tuner.feature_discovery_parents(models, 10)] == [
        "4_LightGBM_GoldenFeatures",
        "5_LightGBM",
    ]


@pytest.mark.parametrize("train_time", [None, float("nan"), float("inf"), -1])
def test_discovery_ignores_invalid_parent_training_time(train_time):
    tuner = object.__new__(MljarTuner)
    tuner._fairness_metric = None
    tuner._top_models_to_improve = 3
    models = [parent("1_LightGBM", 1, train_time=train_time)]
    assert tuner.feature_discovery_parents(models, max_train_time=10) == []


def test_best_parents_profile_order_and_names():
    tuner = object.__new__(MljarTuner)
    tuner._fairness_metric = None
    tuner._top_models_to_improve = 3
    tuner._unique_params_keys = []
    tuner._ml_task = "regression"
    models = [
        parent("1_LightGBM", 3),
        parent("2_LightGBM_GoldenFeatures", 1, True),
        parent("3_LightGBM", 2),
        parent("4_LightGBM", 4),
    ]
    parents = tuner.feature_discovery_parents(models)
    assert [p.get_final_loss() for p in parents] == [1, 2, 3]
    configs = {
        p.get_name(): {"profiles": ["default", "compact", "fast"], "folds": {}}
        for p in parents
    }
    candidates = tuner.get_feature_discovery_params(models, parents, configs)
    assert len(candidates) == 9
    assert candidates[0]["name"] == "5_LightGBM_GoldenFeatures_FeatureDiscovery_Default"
    assert [
        p["preprocessing"]["feature_discovery"]["feature_set"] for p in candidates
    ] == ["default"] * 3 + ["compact"] * 3 + ["fast"] * 3
    assert "golden_features" in candidates[0]["preprocessing"]
    assert all("optuna_time_budget" not in p for p in candidates)
    assert all(p["learner"]["learning_rate"] == 0.1 for p in candidates)
    assert all("feature_discovery" not in m.params["preprocessing"] for m in models)


def test_auto_and_explicit_dependency_and_validation_checks(monkeypatch):
    model = AutoML(mode="Compete", verbose=0)
    model._verbose = 0
    model._validation_strategy = {"validation_type": "split"}
    monkeypatch.setattr(
        "supervised.base_automl.check_discovery_dependencies",
        Mock(side_effect=AutoMLException("install extra")),
    )
    assert model._get_feature_discovery() is False
    model.feature_discovery = True
    with pytest.raises(AutoMLException, match="install extra"):
        model._get_feature_discovery()
    monkeypatch.setattr(
        "supervised.base_automl.check_discovery_dependencies", lambda: None
    )
    assert model._get_feature_discovery() is True
    model._validation_strategy = {"validation_type": "custom"}
    with pytest.raises(AutoMLException, match="custom/group/time"):
        model._get_feature_discovery()
    model.feature_discovery = "auto"
    assert model._get_feature_discovery() is False
    model.feature_discovery = {}
    with pytest.raises(ValueError, match="feature_discovery"):
        model._get_feature_discovery()
