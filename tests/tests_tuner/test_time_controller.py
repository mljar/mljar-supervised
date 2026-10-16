import time
import unittest

import pytest
from numpy.testing import assert_almost_equal

from supervised.tuner.time_controller import TimeController


def test_known_training_cost_cannot_consume_next_steps_budget(monkeypatch):
    tc = TimeController(
        0,
        700,
        None,
        ["default_algorithms", "not_so_random", "feature_discovery"],
        ["LightGBM"],
    )
    tc.log_time("1_LightGBM", "LightGBM", "default_algorithms", 20)
    # Five seconds remain in tuning, but 55 remain in the overall run.
    monkeypatch.setattr("supervised.tuner.time_controller.time.time", lambda: 645)
    assert tc.remaining_step_time("not_so_random") == pytest.approx(5)
    assert not tc.enough_time("LightGBM", "not_so_random")
    assert tc.enough_time("LightGBM", "feature_discovery")


def test_model_admission_counts_unlogged_wall_time(monkeypatch):
    tc = TimeController(0, 100, None, ["default_algorithms"], ["LightGBM"])
    tc.log_time("1_LightGBM", "LightGBM", "default_algorithms", 20)
    monkeypatch.setattr("supervised.tuner.time_controller.time.time", lambda: 95)
    assert not tc.enough_time("LightGBM", "default_algorithms")


def test_discovery_uses_parent_cost_instead_of_algorithm_average(monkeypatch):
    tc = TimeController(
        0, 350, None, ["default_algorithms", "feature_discovery"], ["LightGBM"]
    )
    tc.log_time("1_LightGBM", "LightGBM", "default_algorithms", 200)
    tc.log_time("2_LightGBM", "LightGBM", "default_algorithms", 10)
    monkeypatch.setattr("supervised.tuner.time_controller.time.time", lambda: 330)
    assert not tc.enough_time("LightGBM", "feature_discovery")
    assert tc.enough_time("LightGBM", "feature_discovery", expected_time=10)
    assert not tc.enough_time("LightGBM", "feature_discovery", expected_time=21)
    # Recheck as other candidates consume the remaining time.
    monkeypatch.setattr("supervised.tuner.time_controller.time.time", lambda: 345)
    assert not tc.enough_time("LightGBM", "feature_discovery", expected_time=10)


def test_unused_time_is_available_to_later_steps(monkeypatch):
    tc = TimeController(
        0, 350, None, ["default_algorithms", "feature_discovery"], ["LightGBM"]
    )
    tc.log_time("1_LightGBM", "LightGBM", "default_algorithms", 80)
    monkeypatch.setattr("supervised.tuner.time_controller.time.time", lambda: 100)
    assert tc.time_should_use("feature_discovery") == pytest.approx(50)
    assert tc.remaining_step_time("feature_discovery") == pytest.approx(250)
    assert tc.enough_time("LightGBM", "feature_discovery")


@pytest.mark.parametrize("limits", [(None, None), (100, 10)])
def test_model_or_unlimited_budget_still_admits_candidates(monkeypatch, limits):
    tc = TimeController(0, *limits, ["default_algorithms"], ["LightGBM"])
    tc.log_time("1_LightGBM", "LightGBM", "default_algorithms", 20)
    monkeypatch.setattr("supervised.tuner.time_controller.time.time", lambda: 150)
    assert tc.enough_time("LightGBM", "default_algorithms")


def test_unknown_algorithm_can_still_get_its_first_trial(monkeypatch):
    tc = TimeController(0, 100, None, ["default_algorithms"], ["LightGBM"])
    monkeypatch.setattr("supervised.tuner.time_controller.time.time", lambda: 95)
    assert tc.enough_time("LightGBM", "default_algorithms")


class TimeControllerTest(unittest.TestCase):
    def test_to_and_from_json(self):
        tc = TimeController(
            start_time=time.time(),
            total_time_limit=10,
            model_time_limit=None,
            steps=["simple_algorithms"],
            algorithms=["Baseline"],
        )
        tc.log_time("1_Baseline", "Baseline", "simple_algorithms", 123.1)

        tc2 = TimeController.from_json(tc.to_json())

        assert_almost_equal(tc2.step_spend("simple_algorithms"), 123.1)
        assert_almost_equal(tc2.model_spend("Baseline"), 123.1)

    def test_enough_time_for_stacking(self):
        for t in [5, 10, 20]:
            tc = TimeController(
                start_time=time.time(),
                total_time_limit=100,
                model_time_limit=None,
                steps=[
                    "default_algorithms",
                    "not_so_random",
                    "golden_features",
                    "insert_random_feature",
                    "features_selection",
                    "hill_climbing_1",
                    "hill_climbing_3",
                    "hill_climbing_5",
                    "ensemble",
                    "stack",
                    "ensemble_stacked",
                ],
                algorithms=["Xgboost"],
            )
            tc.log_time("1_Xgboost", "Xgboost", "default_algorithms", t)
            tc.log_time("2_Xgboost", "Xgboost", "not_so_random", t)
            tc.log_time("3_Xgboost", "Xgboost", "insert_random_feature", t)
            tc.log_time("4_Xgboost", "Xgboost", "features_selection", t)
            tc.log_time("5_Xgboost", "Xgboost", "hill_climbing_1", t)
            tc.log_time("6_Xgboost", "Xgboost", "hill_climbing_2", t)
            tc.log_time("7_Xgboost", "Xgboost", "hill_climbing_3", t)

            tc._start_time = time.time() - 7 * t
            assert_almost_equal(tc.already_spend(), 7 * t)
            if t < 20:
                self.assertTrue(tc.enough_time("Xgboost", "stack"))
            else:
                self.assertFalse(tc.enough_time("Xgboost", "stack"))
            self.assertTrue(tc.enough_time("Ensemble_Stacked", "ensemble_stacked"))
