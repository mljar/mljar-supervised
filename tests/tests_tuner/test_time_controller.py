import os
import time
import unittest
from numpy.testing import assert_almost_equal
from supervised.tuner.time_controller import TimeController


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
