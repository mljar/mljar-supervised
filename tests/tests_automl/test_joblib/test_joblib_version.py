import unittest
import joblib
import os
import numpy as np
import pandas as pd
import shutil
import tempfile
from supervised import AutoML
from supervised.model_framework import ModelFramework


class TestModelFramework(unittest.TestCase):

    automl_dir = "automl_testing"
    results_path = "results"
    model_subpath = "framework.json"

    def tearDown(self):
        shutil.rmtree(self.automl_dir, ignore_errors=True)

    def test_load_joblib_version(self):
        # Test version 1.0.0
        jb_version = "1.0.0"
        joblib.__version__ = jb_version

        # Create and fit AutoML
        X = np.random.uniform(size=(60, 2))
        y = np.random.randint(0, 2, size=(60,))
        automl = AutoML(
            results_path=self.results_path,
            model_time_limit=10,
            algorithms=["Xgboost"],
            mode="Compete",
            explain_level=0,
            start_random_models=1,
            hill_climbing_steps=0,
            top_models_to_improve=0,
            kmeans_features=False,
            golden_features=False,
            features_selection=False,
            boost_on_errors=False,
        )
        automl.fit(X, y)

        expected_result = jb_version
        actual_result = ModelFramework.load(self.results_path, self.model_subpath)
        self.assertEqual(expected_result, actual_result)

        # Test version 2.0.0
        jb_version = "2.0.0"
        joblib.__version__ = jb_version

        expected_result = "Different version"
        actual_result = ModelFramework.load(self.results_path, self.model_subpath)
        self.assertEqual(expected_result, actual_result)

        # Test version None
        joblib.__version__ = None

        expected_result = "No version found"
        actual_result = ModelFramework.load(self.results_path, self.model_subpath)
        self.assertEqual(expected_result, actual_result)


if __name__ == '__main__':
    unittest.main()
