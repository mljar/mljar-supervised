import copy
import warnings
from unittest.mock import patch

import pandas as pd
import pytest

from supervised.algorithms.registry import AlgorithmsRegistry
from supervised.tuner.mljar_tuner import MljarTuner


def tuner(**kwargs):
    params = dict(
        tuner_params={
            "start_random_models": 3,
            "hill_climbing_steps": 1,
            "top_models_to_improve": 1,
        },
        algorithms=["Kumo"],
        ml_task="binary_classification",
        eval_metric="auc",
        validation_strategy={},
        explain_level=1,
        data_info={"columns_info": {}, "target_info": [], "rows": 100, "cols": 5},
        golden_features=False,
        features_selection=False,
        train_ensemble=False,
        stack_models=False,
        adjust_validation=False,
        boost_on_errors=False,
        kmeans_features=False,
        mix_encoding=False,
        optuna_time_budget=None,
        optuna_init_params={},
        optuna_verbose=False,
        n_jobs=2,
        seed=42,
    )
    params.update(kwargs)
    return MljarTuner(**params)


def test_default_and_random_scheduling():
    t = tuner(custom_params={"kumo": {"device": "cpu", "max_context_size": 64}})
    before = copy.deepcopy(
        AlgorithmsRegistry.registry["binary_classification"]["Kumo"]["default_params"]
    )
    default = t.default_params(0)
    assert len(default) == 1
    assert default[0]["learner"]["max_context_size"] == 64
    assert default[0]["learner"]["eval_metric_name"] == "auc"
    assert default[0]["learner"]["num_estimators"] == 8
    random = t.get_not_so_random_params(1, [])
    assert len(random) == 2
    assert all(p["learner"]["max_context_size"] == 64 for p in random)
    assert (
        AlgorithmsRegistry.registry["binary_classification"]["Kumo"]["default_params"]
        == before
    )
    assert "scale" not in str(default[0]["preprocessing"])


def test_weighted_candidates_are_not_scheduled_and_warn_once():
    t = tuner(data_info={"is_sample_weighted": True})
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        assert t._get_model_params("Kumo", 1) is None
        assert t._get_model_params("Kumo", 2) is None
    assert len(captured) == 1
    assert "sample-weighted" in str(captured[0].message)


def test_boost_on_errors_excludes_kumo():
    t = tuner()
    with patch.object(
        t,
        "df_models_algorithms",
        return_value=(pd.DataFrame({"model_type": ["Kumo"]}), ["Kumo"]),
    ):
        assert t.boost_params([], "/unused", 60) == []


@pytest.mark.parametrize(
    "method",
    [
        "fairness_optimization_binary_classification",
        "fairness_optimization_multiclass_classification",
        "fairness_optimization_regression",
    ],
)
def test_fairness_never_reweights_kumo(method):
    # Exclude before fairness weights/model-specific methods are consulted.
    from types import SimpleNamespace

    t = tuner()
    m = SimpleNamespace(params={"preprocessing": {}})
    data = pd.DataFrame({"model_type": ["Kumo"], "model": [m]})
    with patch.object(t, "df_models_algorithms", return_value=(data, ["Kumo"])), patch(
        "supervised.tuner.mljar_tuner.load_data",
        side_effect=[pd.DataFrame({"group": [0, 1]}), pd.DataFrame({"target": [0, 1]})],
    ):
        with pytest.warns(UserWarning, match="sample-weighted"):
            assert getattr(t, method)([], "/unused") == []


def test_tabiclv2_scheduling_and_weights():
    t = tuner(
        algorithms=["TabICLv2"], custom_params={"tabiclv2": {"max_context_size": 128}}
    )
    params = t.default_params(0)
    assert len(params) == 1
    assert params[0]["learner"]["model_type"] == "TabICLv2"
    assert params[0]["learner"]["max_context_size"] == 128
    assert len(t.get_not_so_random_params(1, [])) == 2
    t = tuner(algorithms=["TabICLv2"], data_info={"is_sample_weighted": True})
    with pytest.warns(UserWarning, match="sample-weighted"):
        assert t._get_model_params("TabICLv2", 1) is None


def test_kumo_sizes_search_and_fixed_override():
    info = AlgorithmsRegistry.registry["binary_classification"]["Kumo"]
    assert info["params"]["size"] == ["small", "medium", "large"]
    assert info["default_params"]["size"] == "small"
    t = tuner(custom_params={"kumo": {"size": "large"}})
    assert t.default_params(0)[0]["learner"]["size"] == "large"
    assert all(
        p["learner"]["size"] == "large" for p in t.get_not_so_random_params(1, [])
    )
