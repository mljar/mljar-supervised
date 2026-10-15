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


@pytest.mark.parametrize("algorithm", ["Kumo", "TabICLv2"])
@pytest.mark.parametrize("precision", [None, "fp16", "fp32"])
def test_initial_precision_and_fixed_overrides(algorithm, precision):
    custom = {algorithm: {"precision": precision}} if precision else {}
    t = tuner(algorithms=[algorithm], custom_params=custom)
    assert t.default_params(0)[0]["learner"]["precision"] == (precision or "fp16")
    for seed in range(5):
        assert t._get_model_params(algorithm, seed)["learner"]["precision"] == (
            precision or "fp16"
        )


@pytest.mark.parametrize("algorithm", ["Kumo", "TabICLv2"])
@pytest.mark.parametrize("precision", [None, "fp16", "fp32"])
@pytest.mark.parametrize("available", [True, False])
def test_precision_hill_climbing_admission_and_deduplication(
    algorithm, precision, available
):
    from types import SimpleNamespace
    from unittest.mock import Mock

    custom = {algorithm: {"precision": precision}} if precision else {}
    t = tuner(algorithms=[algorithm], custom_params=custom)
    params = t.default_params(0)[0]
    original = copy.deepcopy(params)
    fold = SimpleNamespace(can_try_fp32=Mock(return_value=available))
    model = SimpleNamespace(params=params, get_name=lambda: "1_Model", learners=[fold])
    frame = pd.DataFrame({"model_type": [algorithm], "model": [model]})
    t._unique_params_keys.append(t.get_params_key(params))
    # Generic hill climbing may also propose FP32; schedule only one copy.
    proposal = dict(params["learner"], precision="fp32")
    with patch.object(
        t, "df_models_algorithms", return_value=(frame, [algorithm])
    ), patch("supervised.tuner.mljar_tuner.HillClimbing.get", return_value=[proposal]):
        candidates = t.get_hill_climbing_params([model])
        if available and precision is None:
            assert len(candidates) == 1
            assert candidates[0]["learner"] == dict(params["learner"], precision="fp32")
            t._unique_params_keys.append(t.get_params_key(candidates[0]))
            assert not t.get_hill_climbing_params([model])
        else:
            assert candidates == []
    assert params == original


def test_precision_trial_checks_every_fold_and_skips_missing_metadata():
    from types import SimpleNamespace
    from unittest.mock import Mock

    t = tuner()
    params = t.default_params(0)[0]
    model = SimpleNamespace(params=params, get_name=lambda: "1_Kumo")
    frame = pd.DataFrame({"model_type": ["Kumo"], "model": [model]})
    with patch.object(t, "df_models_algorithms", return_value=(frame, ["Kumo"])), patch(
        "supervised.tuner.mljar_tuner.HillClimbing.get",
        side_effect=lambda *a: [None, None],
    ):
        assert t.get_hill_climbing_params([model]) == []
        model.learners = [
            SimpleNamespace(can_try_fp32=Mock(return_value=True)),
            SimpleNamespace(can_try_fp32=Mock(return_value=False)),
        ]
        assert t.get_hill_climbing_params([model]) == []
        for fold in model.learners:
            fold.can_try_fp32.assert_called_once()
