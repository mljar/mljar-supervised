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
        self.assertIsInstance(p, np.ndarray)
        self.assertEqual(len(p), X.shape[0])

    def test_one_column_input_regression(self):
        a = AutoML(
            results_path=self.automl_dir,
            total_time_limit=5,
            explain_level=0,
            start_random_models=1,
        )

        X, y = datasets.make_regression(n_features=1)

        a.fit(X, y)
        p = a.predict(X)

        self.assertIsInstance(p, np.ndarray)
        self.assertEqual(len(p), X.shape[0])

    def test_one_column_input_bin_class(self):
        a = AutoML(
            results_path=self.automl_dir,
            total_time_limit=5,
            explain_level=0,
            start_random_models=1,
        )

        X = pd.DataFrame({"feature_1": np.random.rand(100)})
        y = (np.random.rand(X.shape[0]) > 0.5).astype(int)

        a.fit(X, y)
        p = a.predict(X)

        self.assertIsInstance(p, np.ndarray)
        self.assertEqual(len(p), X.shape[0])

    def test_different_input_types(self):
        """Test the different data input types for AutoML"""
        model = AutoML(
            total_time_limit=10,
            explain_level=0,
            start_random_models=1,
            algorithms=["Linear"],
            verbose=0,
        )
        X, y = datasets.make_regression()

        # First test - X and y as numpy arrays

        pred = model.fit(X, y).predict(X)

        self.assertIsInstance(pred, np.ndarray)
        self.assertEqual(len(pred), X.shape[0])

        del model

        model = AutoML(
            total_time_limit=10,
            explain_level=0,
            start_random_models=1,
            algorithms=["Linear"],
            verbose=0,
        )
        # Second test - X and y as pandas dataframe
        X_pandas = pd.DataFrame(X)
        y_pandas = pd.DataFrame(y)
        pred_pandas = model.fit(X_pandas, y_pandas).predict(X_pandas)

        self.assertIsInstance(pred_pandas, np.ndarray)
        self.assertEqual(len(pred_pandas), X.shape[0])

        del model

        model = AutoML(
            total_time_limit=10,
            explain_level=0,
            start_random_models=1,
            algorithms=["Linear"],
            verbose=0,
        )
        # Third test - X and y as lists
        X_list = pd.DataFrame(X).values.tolist()
        y_list = pd.DataFrame(y).values.tolist()
        pred_list = model.fit(X_pandas, y_pandas).predict(X_pandas)

        self.assertIsInstance(pred_list, np.ndarray)
        self.assertEqual(len(pred_list), X.shape[0])

    def test_integration_float16_data(self):
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
        X = pd.DataFrame(X)
        X = X.astype(float)
        a.fit(X, y)
        p = a.predict(X)
        self.assertIsInstance(p, np.ndarray)
        self.assertEqual(len(p), X.shape[0])
