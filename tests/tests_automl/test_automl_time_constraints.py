import os
import unittest
import tempfile
import json
import numpy as np
import pandas as pd
import shutil
from supervised import AutoML
from numpy.testing import assert_almost_equal
from sklearn import datasets
from supervised.exceptions import AutoMLException


class AutoMLTimeConstraintsTest(unittest.TestCase):

    automl_dir = "automl_tests"

    def tearDown(self):
        shutil.rmtree(self.automl_dir)

    def test_set_total_time_limit(self):
        model_type = "Xgboost"
        automl = AutoML(
            results_path=self.automl_dir, total_time_limit=100, algorithms=[model_type]
        )
        
        automl._time_spend['simple_algorithms'] = 0
        automl._time_spend['default_algorithms'] = 0
        automl._fit_level = 'not_so_random'
        time_spend = 0
        for _ in range(12):
            automl._start_time -= 10
            automl.log_train_time(model_type, 10)
            if automl._enough_time_to_train(model_type):
                time_spend += 10
                
        self.assertTrue(time_spend < 100)

    def test_set_model_time_limit(self):
        model_type = "Xgboost"
        automl = AutoML(
            results_path=self.automl_dir, model_time_limit=10, algorithms=[model_type]
        )
        
        for _ in range(12):
            automl.log_train_time(model_type, 10)
            # should be always true
            self.assertTrue(automl._enough_time_to_train(model_type))

    def test_set_model_time_limit_omit_total_time(self):
        model_type = "Xgboost"
        automl = AutoML(
            results_path=self.automl_dir,
            model_time_limit=10,
            total_time_limit=10,  # this parameter setting should be omitted
            algorithms=[model_type],
        )
        
        for _ in range(12):
            automl.log_train_time(model_type, 10)
            # should be always true
            self.assertTrue(automl._enough_time_to_train(model_type))

    def test_enough_time_to_train(self):
        model_type = "Xgboost"
        model_type_2 = "LightGBM"

        automl = AutoML(
            results_path=self.automl_dir,
            total_time_limit=10,  # this parameter setting should be omitted
            algorithms=[model_type, model_type_2],
        )

        for i in range(5):
            # should be always true
            self.assertTrue(automl._enough_time_to_train(model_type))
            automl.log_train_time(model_type, 1)
