import shutil
import unittest

import numpy as np
import pandas as pd

from supervised import AutoML


class AutoMLDataTypesTest(unittest.TestCase):
    automl_dir = "automl_tests"
    rows = 250

    def tearDown(self):
        shutil.rmtree(self.automl_dir, ignore_errors=True)

    def test_category_data_type(self):
        X = np.random.rand(self.rows, 3)
        X = pd.DataFrame(X, columns=[f"f{i}" for i in range(3)])
        y = np.random.randint(0, 2, self.rows)

        X["f1"] = X["f1"].astype("category")

        automl = AutoML(
            results_path=self.automl_dir,
            total_time_limit=1,
            algorithms=["CatBoost"],
            train_ensemble=False,
            explain_level=0,
            start_random_models=1,
        )
        automl.fit(X, y)

    def test_encoding_strange_characters(self):
        X = np.random.rand(self.rows, 3)
        X = pd.DataFrame(X, columns=[f"f{i}" for i in range(3)])
        y = np.random.permutation(["ɛ", "🂲"] * int(self.rows / 2))

        automl = AutoML(
            results_path=self.automl_dir,
            total_time_limit=1,
            algorithms=["Baseline"],
            train_ensemble=False,
            explain_level=0,
            start_random_models=1,
        )
        automl.fit(X, y)

    def test_numeric_category_target(self):
        """Regression test: target stored as numeric category dtype should not
        raise 'ValueError: pandas dtypes must be int, float or bool'.
        See: https://github.com/mljar/mljar-supervised/issues/XXX
        """
        X = np.random.rand(self.rows, 3)
        X = pd.DataFrame(X, columns=[f"f{i}" for i in range(3)])
        # Binary target with integer values stored as category dtype
        y = pd.Series(np.random.randint(0, 2, self.rows)).astype("category")

        automl = AutoML(
            results_path=self.automl_dir,
            total_time_limit=1,
            algorithms=["Baseline"],
            train_ensemble=False,
            explain_level=0,
            start_random_models=1,
        )
        automl.fit(X, y)

    def test_numeric_category_features_and_target(self):
        """Regression test: both X feature columns and y target stored as
        numeric category dtype must be handled without errors.
        """
        X = pd.DataFrame(
            {
                "f0": np.random.rand(self.rows),
                # Numeric-valued categorical feature — the exact scenario from the issue
                "f1": pd.Series(
                    np.random.randint(1, 5, self.rows)
                ).astype("category"),
                "f2": pd.Series(
                    np.random.randint(0, 3, self.rows)
                ).astype("category"),
            }
        )
        y = pd.Series(np.random.randint(0, 2, self.rows)).astype("category")

        automl = AutoML(
            results_path=self.automl_dir,
            total_time_limit=1,
            algorithms=["Baseline"],
            train_ensemble=False,
            explain_level=0,
            start_random_models=1,
        )
        automl.fit(X, y)

    def test_string_category_target(self):
        """Regression test: target with string-valued category dtype is handled."""
        X = np.random.rand(self.rows, 3)
        X = pd.DataFrame(X, columns=[f"f{i}" for i in range(3)])
        y = pd.Series(
            np.random.choice(["cat", "dog"], size=self.rows)
        ).astype("category")

        automl = AutoML(
            results_path=self.automl_dir,
            total_time_limit=1,
            algorithms=["Baseline"],
            train_ensemble=False,
            explain_level=0,
            start_random_models=1,
        )
        automl.fit(X, y)
