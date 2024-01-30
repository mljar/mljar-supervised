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
