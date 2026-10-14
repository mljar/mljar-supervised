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
    assert combined.time_should_use("golden_features") * 2 == pytest.approx(
        original.time_should_use("golden_features")
    )
    assert combined.time_should_use("feature_discovery") == combined.time_should_use(
        "golden_features"
    )
    assert discovery.time_should_use("feature_discovery") == original.time_should_use(
        "golden_features"
    )
    assert combined.time_should_use("stack") == original.time_should_use("stack")
    restored = TimeController.from_json(combined.to_json())
    assert restored.time_should_use("feature_discovery") == combined.time_should_use(
        "feature_discovery"
    )


def parent(name, score, golden=False):
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
        _is_stacked=False,
        get_name=lambda: name,
        get_type=lambda: "LightGBM",
        get_final_loss=lambda: score,
        get_train_time=lambda: 2,
    )


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
