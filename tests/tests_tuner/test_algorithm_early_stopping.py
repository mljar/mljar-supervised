import copy
import json

import pytest

from supervised.tuner.algorithm_early_stopping import AlgorithmEarlyStopping
from supervised.tuner.time_controller import TimeController
from supervised.utils.metric import F1_METRICS, Metric


def candidate(algorithm="Weak", config=0, **extra):
    return dict(name=f"{algorithm}_{config}", status="initialized", data_type="original",
                learner=dict(model_type=algorithm, depth=config, seed=1, n_jobs=1),
                preprocessing={}, validation_strategy=dict(validation_type="kfold", k_folds=3),
                **extra)


def observe(policy, algorithm, config, cost, step="not_so_random", **changes):
    params = candidate(algorithm, config)
    params.update(changes, status="trained")
    score = cost if policy.minimize else -cost
    policy.observe(step, params, score)
    return params


def history(metric="logloss", scale=1, offset=0, weak=3):
    p = AlgorithmEarlyStopping(metric)
    observe(p, "Reference", 0, 10 * scale + offset, "simple_algorithms")
    observe(p, "Leader", 0, offset, "default_algorithms")
    for i in range(3):
        observe(p, "Weak", i, weak * scale + offset)
    return p


@pytest.mark.parametrize("metric", ["logloss", "rmse", "mse", "mae", "mape", "auc",
                                  "r2", "pearson", "spearman", "accuracy", "acc",
                                  "average_precision", *F1_METRICS, "user_defined_metric"])
def test_one_rule_for_every_supported_metric(metric):
    p = history(metric)
    assert p.decision("not_so_random", candidate())["normalized_gap"] == pytest.approx(.3)
    assert p.decision("hill_climbing_2", candidate())["successful_configurations"] == 3
    assert Metric({"name": metric}).get_minimize_direction() == p.minimize


@pytest.mark.parametrize("scale,offset", [(1, 0), (10, 100), (.01, -20), (1e-20, 0), (1, -100)])
def test_shifted_negative_and_rescaled_custom_losses(scale, offset):
    p = history("user_defined_metric", scale, offset)
    assert p.decision("not_so_random", candidate())["normalized_gap"] == pytest.approx(.3)
    assert history("user_defined_metric", scale, offset, weak=2).decision("not_so_random", candidate()) is None


def test_negative_auc_and_fixed_reference():
    p = AlgorithmEarlyStopping("auc")
    observe(p, "Reference", 0, -.7, "simple_algorithms")
    observe(p, "Leader", 0, -.8, "default_algorithms")
    for i in range(3):
        observe(p, "Weak", i, -.77)
    assert p.decision("not_so_random", candidate())["normalized_gap"] == pytest.approx(.3)
    observe(p, "Leader", 1, -.85)
    assert p.reference == dict(model="Reference_0", loss=-.7)
    assert p.decision("not_so_random", candidate())["normalized_gap"] == pytest.approx(.08/.15)


def test_exploration_counts_distinct_successes_across_stages():
    p = AlgorithmEarlyStopping("logloss")
    observe(p, "Reference", 0, 10, "simple_algorithms")
    observe(p, "Leader", 0, 0, "default_algorithms")
    observe(p, "Weak", 0, 4, "default_algorithms")
    dup = observe(p, "Weak", 0, 3)
    dup["learner"].update(seed=999, n_jobs=10)
    p.observe("not_so_random", dup, 3)
    for status in ("error", "skipped", "initialized"):
        failed = candidate(config=42)
        failed["status"] = status
        p.observe("not_so_random", failed, 8)
    observe(p, "Weak", 1, float("nan"))
    observe(p, "Weak", 1, float("inf"))
    observe(p, "Weak", 1, 3)
    assert p.decision("hill_climbing_1", candidate()) is None
    observe(p, "Weak", 2, 3, "hill_climbing_1")
    assert p.decision("hill_climbing_2", candidate()) is not None


@pytest.mark.parametrize("stage", ["adjust_validation", "default_algorithms", "simple_algorithms",
                                  "mix_encoding", "golden_features", "feature_discovery",
                                  "features_selection", "boost_on_errors", "stack", "ensemble"])
def test_only_original_random_and_hill_climbing_candidates_are_paused(stage):
    assert history().decision(stage, candidate()) is None


@pytest.mark.parametrize("change", [
    dict(data_type="original_features_selection"), dict(is_stacked=True),
    dict(name="Weak_RandomFeature"), dict(preprocessing={"drop_features": ["x"]}),
    dict(preprocessing={"golden_features": {"count": 10}}),
    dict(validation_strategy={"validation_type": "kfold", "k_folds": 5}),
    dict(validation_strategy={"validation_type": "kfold", "k_folds": 3, "sample_weight_path": "weights.data"}),
])
def test_different_representations_weights_and_splits_are_independent(change):
    p = history()
    params = candidate()
    params.update(change, status="trained")
    assert p.decision("hill_climbing_1", params) is None
    before = p.to_json()
    p.observe("hill_climbing_1", params, -100)
    assert p.to_json() == before


def test_adjustment_and_probe_do_not_become_reference():
    p = AlgorithmEarlyStopping("logloss")
    observe(p, "Reference", 0, 100, "adjust_validation")
    observe(p, "Probe", 0, 100, name="Probe_RandomFeature")
    assert p.reference is None
    observe(p, "Real", 0, 10, "simple_algorithms")
    assert p.reference["model"] == "Real_0"


def test_effective_custom_configuration_and_partial_exploration_survive_resume():
    p = AlgorithmEarlyStopping("user_defined_metric")
    observe(p, "Reference", 0, 10, "simple_algorithms")
    observe(p, "Leader", 0, 0, "default_algorithms")
    for i in range(4):
        observe(p, "Weak", i, 3, custom_params={"depth": 42})
    assert len(p.algorithms["Weak"]["configurations"]) == 1
    p = AlgorithmEarlyStopping.from_json(json.loads(json.dumps(p.to_json())))
    observe(p, "Weak", 4, 3)
    assert p.decision("not_so_random", candidate()) is None
    observe(p, "Weak", 5, 3)
    assert p.decision("hill_climbing_1", candidate()) is not None


def test_no_improvement_and_disabled_policy_keep_tuning():
    p = AlgorithmEarlyStopping("logloss")
    for i in range(3):
        observe(p, "Weak", i, 10)
    assert p.decision("not_so_random", candidate()) is None
    p = history()
    p.enabled = False
    assert p.decision("not_so_random", candidate()) is None


def test_queued_candidates_are_rechecked_and_resume_does_not_repeat_messages():
    p = history(weak=1)
    params = candidate(config=4)
    assert not p.skip_candidate("not_so_random", params)
    observe(p, "Leader", 1, -5)
    assert p.skip_candidate("not_so_random", params)
    assert params["status"] == "skipped" and params["final_loss"] is None
    assert params["train_time"] is None
    assert len(p.messages()) == 1
    reloaded = AlgorithmEarlyStopping.from_json(json.loads(json.dumps(p.to_json(), allow_nan=False)))
    assert reloaded.messages() == []
    assert reloaded.skip_candidate("hill_climbing_1", candidate(config=5))
    assert reloaded.messages() == []
    assert len(reloaded.skipped) == 2
    assert len(reloaded.algorithms["Weak"]["configurations"]) == 3


def test_generation_filter_preserves_skip_evidence_and_completed_candidates():
    p = history()
    pending = candidate(config=4)
    trained = candidate(config=5)
    trained["status"] = "trained"
    selected = candidate(config=6)
    selected["data_type"] = "original_features_selection"
    original = copy.deepcopy(trained)
    assert p.filter_candidates("not_so_random", [pending, trained, selected]) == [pending, trained, selected]
    assert pending["status"] == "skipped"
    assert trained == original
    assert selected["status"] == "initialized"


def test_tuning_allowance_uses_active_algorithms_and_remaining_wall_time(monkeypatch):
    tc = TimeController(0, 650, None, ["default_algorithms", "not_so_random"], ["LightGBM", "Extra Trees"])
    monkeypatch.setattr("supervised.tuner.time_controller.time.time", lambda: 300)
    tc.set_active_algorithms("not_so_random", ["LightGBM", "Extra Trees"])
    assert tc.learner_time_limit("LightGBM", "not_so_random", 5) == pytest.approx(35)
    tc.set_active_algorithms("not_so_random", ["LightGBM"])
    assert tc.learner_time_limit("LightGBM", "not_so_random", 5) == pytest.approx(70)
    assert tc.learner_time_limit("Extra Trees", "not_so_random", 5) == 0
    monkeypatch.setattr("supervised.tuner.time_controller.time.time", lambda: 640)
    assert tc.learner_time_limit("LightGBM", "not_so_random", 5) == pytest.approx(2)
    assert tc.learner_time_limit("LightGBM", "default_algorithms", 5) is None
