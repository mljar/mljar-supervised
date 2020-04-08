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
        automl._estimate_training_times()
        time_spend = 0
        for _ in range(12):
            automl.log_train_time(model_type, 10)
            if automl._enough_time_to_train(model_type):
                time_spend += 10

        self.assertTrue(time_spend < 100)

    def test_set_model_time_limit(self):
        model_type = "Xgboost"
        automl = AutoML(
            results_path=self.automl_dir, model_time_limit=10, algorithms=[model_type]
        )
        automl._estimate_training_times()
        print(automl._time_limit)
        for _ in range(12):
            automl.log_train_time(model_type, 10)
            # should be always true
            self.assertTrue(automl._enough_time_to_train(model_type))

    def test_set_model_time_limit_omit_total_time(self):
        model_type = "Xgboost"
        automl = AutoML(
            results_path=self.automl_dir, model_time_limit=10, 
            total_time_limit=10, # this parameter setting should be omitted
            algorithms=[model_type]
        )
        automl._estimate_training_times()
        print(automl._time_limit)
        for _ in range(12):
            automl.log_train_time(model_type, 10)
            # should be always true
            self.assertTrue(automl._enough_time_to_train(model_type))
    