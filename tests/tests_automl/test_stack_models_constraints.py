import os
import unittest
import tempfile
import json
import numpy as np
import pandas as pd
import shutil

from supervised import AutoML
from supervised.exceptions import AutoMLException


class AutoMLStackModelsConstraintsTest(unittest.TestCase):

    automl_dir = "automl_testing"

    def tearDown(self):
        shutil.rmtree(self.automl_dir, ignore_errors=True)

    def test_allow_stack_models(self):

        X = np.random.uniform(size=(100, 2))
        y = np.random.randint(0, 2, size=(100,))
        X[:,0] = y
        X[:,1] = -y

        automl = AutoML(results_path=self.automl_dir, 
                    total_time_limit=5,
                    mode="Compete",
                    validation_strategy={
                        "validation_type": "kfold",
                        "k_folds": 5
                    })
        automl.fit(X, y)
        self.assertTrue(automl._stack_models)
        self.assertTrue(automl.tuner._stack_models)
        self.assertTrue(automl._time_ctrl._is_stacking)
        
    def test_disable_stack_models(self):

        X = np.random.uniform(size=(100, 2))
        y = np.random.randint(0, 2, size=(100,))
        X[:,0] = y
        X[:,1] = -y

        automl = AutoML(results_path=self.automl_dir, 
                    total_time_limit=5,
                    mode="Compete",
                    validation_strategy={
                        "validation_type": "split",
                     })
        automl.fit(X, y)
        self.assertFalse(automl._stack_models)
        self.assertFalse(automl.tuner._stack_models)
        self.assertFalse(automl._time_ctrl._is_stacking)
        
    def test_disable_stack_models_adjusted_validation(self):

        X = np.random.uniform(size=(100, 2))
        y = np.random.randint(0, 2, size=(100,))
        X[:,0] = y
        X[:,1] = -y

        automl = AutoML(results_path=self.automl_dir, 
                    total_time_limit=5,
                    mode="Compete")
        automl.fit(X, y)
        # the stacking should be disabled
        # because of small time limit
        self.assertFalse(automl._stack_models)
        self.assertFalse(automl.tuner._stack_models)
        self.assertFalse(automl._time_ctrl._is_stacking)
    