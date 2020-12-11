import os
import unittest
import tempfile
import json
import time
import numpy as np
import pandas as pd
import shutil
from supervised import AutoML
from numpy.testing import assert_almost_equal
from sklearn import datasets
from supervised.exceptions import AutoMLException
from supervised.tuner.time_controller import TimeController


class AutoMLTimeConstraintsTest(unittest.TestCase):

    automl_dir = "automl_tests"

    def tearDown(self):
        shutil.rmtree(self.automl_dir, ignore_errors=True)

    def test_set_total_time_limit(self):
        model_type = "Xgboost"
        automl = AutoML(
            results_path=self.automl_dir, total_time_limit=100, algorithms=[model_type]
        )

        automl._time_ctrl = TimeController(
            time.time(), 100, None, ["simple_algorithms", "not_so_random"], "Xgboost"
        )

        time_spend = 0
        for i in range(12):
            automl._start_time -= 10
            automl._time_ctrl.log_time(f"Xgboost_{i}", model_type, "not_so_random", 10)
            if automl._time_ctrl.enough_time(model_type, "not_so_random"):
                time_spend += 10

        self.assertTrue(time_spend < 100)

    def test_set_model_time_limit(self):
        model_type = "Xgboost"
        automl = AutoML(
            results_path=self.automl_dir, model_time_limit=10, algorithms=[model_type]
        )
        automl._time_ctrl = TimeController(
            time.time(), None, 10, ["simple_algorithms", "not_so_random"], "Xgboost"
        )

        for i in range(12):
            automl._time_ctrl.log_time(f"Xgboost_{i}", model_type, "not_so_random", 10)
            # should be always true
            self.assertTrue(automl._time_ctrl.enough_time(model_type, "not_so_random"))

    def test_set_model_time_limit_omit_total_time(self):
        model_type = "Xgboost"
        automl = AutoML(
            results_path=self.automl_dir,
            total_time_limit=10,
            model_time_limit=10,
            algorithms=[model_type],
        )
        automl._time_ctrl = TimeController(
            time.time(), 10, 10, ["simple_algorithms", "not_so_random"], "Xgboost"
        )

        for i in range(12):
            automl._time_ctrl.log_time(f"Xgboost_{i}", model_type, "not_so_random", 10)
            # should be always true
            self.assertTrue(automl._time_ctrl.enough_time(model_type, "not_so_random"))

    def test_enough_time_to_train(self):
        model_type = "Xgboost"
        model_type_2 = "LightGBM"

        model_type = "Xgboost"
        automl = AutoML(
            results_path=self.automl_dir,
            total_time_limit=10,
            model_time_limit=10,
            algorithms=[model_type, model_type_2],
        )
        automl._time_ctrl = TimeController(
            time.time(),
            10,
            10,
            ["simple_algorithms", "not_so_random"],
            [model_type, model_type_2],
        )

        for i in range(5):
            automl._time_ctrl.log_time(f"Xgboost_{i}", model_type, "not_so_random", 1)
            # should be always true
            self.assertTrue(automl._time_ctrl.enough_time(model_type, "not_so_random"))

        for i in range(5):
            automl._time_ctrl.log_time(
                f"LightGBM_{i}", model_type_2, "not_so_random", 1
            )
            # should be always true
            self.assertTrue(
                automl._time_ctrl.enough_time(model_type_2, "not_so_random")
            )

    def test_expected_learners_cnt(self):
        automl = AutoML(results_path=self.automl_dir)
        automl._validation_strategy = {"k_folds": 7, "repeats": 6}
        self.assertEqual(automl._expected_learners_cnt(), 42)

        automl._validation_strategy = {"k_folds": 7}
        self.assertEqual(automl._expected_learners_cnt(), 7)
        automl._validation_strategy = {}
        self.assertEqual(automl._expected_learners_cnt(), 1)
