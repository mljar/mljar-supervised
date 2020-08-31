import os
import unittest
import tempfile
import json
import numpy as np
import pandas as pd
import shutil
from sklearn import datasets

from supervised import AutoML


class AutoMLIntegrationTest(unittest.TestCase):

    automl_dir = "automl_1"

    def tearDown(self):
        shutil.rmtree(self.automl_dir, ignore_errors=True)

    def test_integration(self):
        a = AutoML(
            results_path=self.automl_dir,
            total_time_limit=1,
            explain_level=0,
            start_random_models=1,
        )

        X, y = datasets.make_classification(
            n_samples=100,
            n_features=5,
            n_informative=4,
            n_redundant=1,
            n_classes=2,
            n_clusters_per_class=3,
            n_repeated=0,
            shuffle=False,
            random_state=0,
        )

        a.fit(X, y)
        p = a.predict(X)
        self.assertTrue(isinstance(p, np.ndarray))

    def test_one_column_input_regression(self):
        a = AutoML(
            results_path=self.automl_dir,
            total_time_limit=5,
            explain_level=0,
            start_random_models=1,
        )

        X = pd.DataFrame({"feature_1": np.random.rand(100)})
        y = np.random.rand(100)

        a.fit(X, y)
        p = a.predict(X)

        self.assertTrue(isinstance(p, np.ndarray))
        self.assertTrue(p.shape[0] == 100)

    def test_one_column_input_bin_class(self):
        a = AutoML(
            results_path=self.automl_dir,
            total_time_limit=5,
            explain_level=0,
            start_random_models=1,
        )

        X = pd.DataFrame({"feature_1": np.random.rand(100)})
        y = (np.random.rand(100) > 0.5).astype(int)

        a.fit(X, y)
        p = a.predict(X)

        self.assertTrue(isinstance(p, np.ndarray))
        self.assertTrue(p.shape[0] == 100)
