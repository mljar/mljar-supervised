import os
import unittest
import json
import numpy as np
import pandas as pd
import shutil
from supervised import AutoML
from numpy.testing import assert_almost_equal
from sklearn import datasets
from supervised.exceptions import AutoMLException

from supervised.algorithms.xgboost import additional

additional["max_rounds"] = 1


class AutoMLTargetsTest(unittest.TestCase):

    automl_dir = "automl_tests"
    rows = 50

    def tearDown(self):
        shutil.rmtree(self.automl_dir, ignore_errors=True)

    def test_bin_class_01(self):
        X = np.random.rand(self.rows, 3)
        X = pd.DataFrame(X, columns=[f"f{i}" for i in range(3)])
        y = np.random.randint(0, 2, self.rows)

        automl = AutoML(
            results_path=self.automl_dir,
            total_time_limit=1,
            algorithms=["Xgboost"],
            train_ensemble=False,
            explain_level=0,
            start_random_models=1,
        )
        automl.fit(X, y)
        pred = automl.predict(X)

        u = np.unique(pred)
        self.assertTrue(0 in u or 1 in u)
        self.assertTrue(len(u) <= 2)

    def test_bin_class_11(self):
        X = np.random.rand(self.rows, 3)
        X = pd.DataFrame(X, columns=[f"f{i}" for i in range(3)])
        y = np.random.randint(0, 2, self.rows) * 2 - 1

        automl = AutoML(
            results_path=self.automl_dir,
            total_time_limit=1,
            algorithms=["Xgboost"],
            train_ensemble=False,
            explain_level=0,
            start_random_models=1,
        )
        automl.fit(X, y)
        p = automl.predict(X)
        pred = automl.predict(X)

        u = np.unique(pred)

        self.assertTrue(-1 in u or 1 in u)
        self.assertTrue(0 not in u)
        self.assertTrue(len(u) <= 2)

    def test_bin_class_AB(self):
        X = np.random.rand(self.rows, 3)
        X = pd.DataFrame(X, columns=[f"f{i}" for i in range(3)])
        y = np.random.permutation(["a", "B"] * int(self.rows / 2))

        automl = AutoML(
            results_path=self.automl_dir,
            total_time_limit=1,
            algorithms=["Xgboost"],
            train_ensemble=False,
            explain_level=0,
            start_random_models=1,
        )
        automl.fit(X, y)
        p = automl.predict(X)
        pred = automl.predict(X)
        u = np.unique(pred)
        self.assertTrue("a" in u or "B" in u)
        self.assertTrue(len(u) <= 2)

    def test_bin_class_AB_missing_targets(self):
        X = np.random.rand(self.rows, 3)
        X = pd.DataFrame(X, columns=[f"f{i}" for i in range(3)])
        y = pd.Series(
            np.random.permutation(["a", "B"] * int(self.rows / 2)), name="target"
        )

        y.iloc[1] = None
        y.iloc[3] = np.NaN
        y.iloc[13] = np.nan

        automl = AutoML(
            results_path=self.automl_dir,
            total_time_limit=1,
            algorithms=["Xgboost"],
            train_ensemble=False,
            explain_level=0,
            start_random_models=1,
        )
        automl.fit(X, y)
        p = automl.predict(X)
        pred = automl.predict(X)

        u = np.unique(pred)
        self.assertTrue("a" in u or "B" in u)
        self.assertTrue(len(u) <= 2)

    def test_multi_class_0123_floats(self):
        X = np.random.rand(self.rows * 4, 3)
        X = pd.DataFrame(X, columns=[f"f{i}" for i in range(3)])
        y = np.random.randint(0, 4, self.rows * 4)
        y = y.astype(float)

        automl = AutoML(
            results_path=self.automl_dir,
            total_time_limit=1,
            algorithms=["Xgboost"],
            train_ensemble=False,
            explain_level=0,
            start_random_models=1,
        )
        automl.fit(X, y)
        pred = automl.predict(X)

        u = np.unique(pred)

        self.assertTrue(0.0 in u or 1.0 in u or 2.0 in u or 3.0 in u)
        self.assertTrue(len(u) <= 4)

    def test_multi_class_0123(self):
        X = np.random.rand(self.rows * 4, 3)
        X = pd.DataFrame(X, columns=[f"f{i}" for i in range(3)])
        y = np.random.randint(0, 4, self.rows * 4)

        automl = AutoML(
            results_path=self.automl_dir,
            total_time_limit=1,
            algorithms=["Xgboost"],
            train_ensemble=False,
            explain_level=0,
            start_random_models=1,
        )
        automl.fit(X, y)
        pred = automl.predict(X)

        u = np.unique(pred)

        self.assertTrue(0 in u or 1 in u or 2 in u or 3 in u)
        self.assertTrue(len(u) <= 4)

    def test_multi_class_0123_strings(self):
        X = np.random.rand(self.rows * 4, 3)
        X = pd.DataFrame(X, columns=[f"f{i}" for i in range(3)])
        y = np.random.randint(0, 4, self.rows * 4)
        y = y.astype(str)

        automl = AutoML(
            results_path=self.automl_dir,
            total_time_limit=1,
            algorithms=["Xgboost"],
            train_ensemble=False,
            explain_level=0,
            start_random_models=1,
        )
        automl.fit(X, y)
        pred = automl.predict(X)

        u = np.unique(pred)

        self.assertTrue("0" in u or "1" in u or "2" in u or "3" in u)
        self.assertTrue(len(u) <= 4)

    def test_multi_class_abcd(self):
        X = np.random.rand(self.rows * 4, 3)
        X = pd.DataFrame(X, columns=[f"f{i}" for i in range(3)])
        y = pd.Series(
            np.random.permutation(["a", "B", "CC", "d"] * self.rows), name="target"
        )

        automl = AutoML(
            results_path=self.automl_dir,
            total_time_limit=1,
            algorithms=["Xgboost"],
            train_ensemble=False,
            explain_level=0,
            start_random_models=1,
        )
        automl.fit(X, y)
        pred = automl.predict(X)

        u = np.unique(pred)

        self.assertTrue(np.intersect1d(u, ["a", "B", "CC", "d"]).shape[0] > 0)
        self.assertTrue(len(u) <= 4)

    def test_multi_class_abcd_np_array(self):
        X = np.random.rand(self.rows * 4, 3)
        X = pd.DataFrame(X, columns=[f"f{i}" for i in range(3)])
        y = np.random.permutation([None, "B", "CC", "d"] * self.rows)

        automl = AutoML(
            results_path=self.automl_dir,
            total_time_limit=1,
            algorithms=["Xgboost"],
            train_ensemble=False,
            explain_level=0,
            start_random_models=1,
        )
        automl.fit(X, y)
        pred = automl.predict(X)

        u = np.unique(pred)

        self.assertTrue(np.intersect1d(u, ["a", "B", "CC", "d"]).shape[0] > 0)
        self.assertTrue(len(u) <= 4)

    def test_multi_class_abcd_mixed_int(self):
        X = np.random.rand(self.rows * 4, 3)
        X = pd.DataFrame(X, columns=[f"f{i}" for i in range(3)])
        y = pd.Series(
            np.random.permutation([1, "B", "CC", "d"] * self.rows), name="target"
        )

        automl = AutoML(
            results_path=self.automl_dir,
            total_time_limit=1,
            algorithms=["Xgboost"],
            train_ensemble=False,
            explain_level=0,
            start_random_models=1,
        )
        automl.fit(X, y)
        pred = automl.predict(X)
        u = np.unique(pred)

        self.assertTrue(np.intersect1d(u, ["a", "B", "CC", "d"]).shape[0] > 0)
        self.assertTrue(len(u) <= 4)

    def test_multi_class_abcd_missing_target(self):
        X = np.random.rand(self.rows * 4, 3)
        X = pd.DataFrame(X, columns=[f"f{i}" for i in range(3)])
        y = pd.Series(
            np.random.permutation(["a", "B", "CC", "d"] * self.rows), name="target"
        )

        y.iloc[0] = None
        y.iloc[1] = None
        automl = AutoML(
            results_path=self.automl_dir,
            total_time_limit=1,
            algorithms=["Xgboost"],
            train_ensemble=False,
            explain_level=0,
            start_random_models=1,
        )
        automl.fit(X, y)
        pred = automl.predict(X)

        u = np.unique(pred)

        self.assertTrue(np.intersect1d(u, ["a", "B", "CC", "d"]).shape[0] > 0)
        self.assertTrue(len(u) <= 4)

    def test_regression(self):
        X = np.random.rand(self.rows, 3)
        X = pd.DataFrame(X, columns=[f"f{i}" for i in range(3)])
        y = np.random.rand(self.rows)

        automl = AutoML(
            results_path=self.automl_dir,
            total_time_limit=1,
            algorithms=["Xgboost"],
            train_ensemble=False,
            explain_level=0,
            start_random_models=1,
        )
        automl.fit(X, y)
        pred = automl.predict(X)

        self.assertIsInstance(pred, np.ndarray)
        self.assertEqual(len(pred), X.shape[0])

    def test_regression_missing_target(self):
        X = np.random.rand(self.rows, 3)
        X = pd.DataFrame(X, columns=[f"f{i}" for i in range(3)])
        y = pd.Series(np.random.rand(self.rows), name="target")

        y.iloc[1] = None

        automl = AutoML(
            results_path=self.automl_dir,
            total_time_limit=1,
            algorithms=["Xgboost"],
            train_ensemble=False,
            explain_level=0,
            start_random_models=1,
        )
        automl.fit(X, y)
        pred = automl.predict(X)

        self.assertIsInstance(pred, np.ndarray)
        self.assertEqual(len(pred), X.shape[0])

    def test_predict_on_empty_dataframe(self):
        X = np.random.rand(self.rows, 3)
        X = pd.DataFrame(X, columns=[f"f{i}" for i in range(3)])
        y = pd.Series(np.random.rand(self.rows), name="target")

        automl = AutoML(
            results_path=self.automl_dir,
            total_time_limit=1,
            algorithms=["Xgboost"],
            train_ensemble=False,
            explain_level=0,
            start_random_models=1,
        )
        automl.fit(X, y)

        with self.assertRaises(AutoMLException) as context:
            pred = automl.predict(pd.DataFrame())

        with self.assertRaises(AutoMLException) as context:
            pred = automl.predict(np.empty(shape=(0, 3)))
