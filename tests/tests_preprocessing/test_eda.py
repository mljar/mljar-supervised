import os
import unittest
import tempfile
import json
import numpy as np
import pandas as pd
import shutil
from sklearn import datasets

from supervised import AutoML
from supervised.preprocessing.eda import EDA


class EDATest(unittest.TestCase):

    automl_dir = "automl_1"

    def tearDown(self):
        shutil.rmtree(self.automl_dir, ignore_errors=True)

    def test_explain_default(self):
        a = AutoML(
            results_path=self.automl_dir,
            total_time_limit=10,
            algorithms=["Random Forest"],
            train_ensemble=False,
            explain_level=2,
        )

        X, y = datasets.make_classification(n_samples=100, n_features=5)

        X = pd.DataFrame(X, columns=[f"f_{i}" for i in range(X.shape[1])])
        y = pd.Series(y, name="class")

        a.fit(X, y)

        result_files = os.listdir(os.path.join(a._results_path, "EDA"))

        for col in X.columns:

            produced = True
            if "".join((col, ".png")) not in result_files:
                produced = False
                break
        self.assertTrue(produced)

        if "target.png" not in result_files:
            produced = False
        self.assertTrue(produced)

        if "README.md" not in result_files:
            produced = False
        self.assertTrue(produced)

        self.tearDown()

    def test_extensive_eda(self):

        X, y = datasets.make_regression(n_samples=100, n_features=5)

        X = pd.DataFrame(X, columns=[f"f_{i}" for i in range(X.shape[1])])
        y = pd.Series(y, name="class")

        results_path = "EDA"
        EDA.extensive_eda(X, y, results_path)
        result_files = os.listdir(results_path)

        for col in X.columns:

            produced = True
            if "_target".join((col, ".png")) not in result_files:
                produced = False
                break
        self.assertTrue(produced)

        X, y = datasets.make_classification(n_samples=100, n_features=5)

        X = pd.DataFrame(X, columns=[f"f_{i}" for i in range(X.shape[1])])
        y = pd.Series(y, name="class")

        results_path = "EDA"
        EDA.extensive_eda(X, y, results_path)
        result_files = os.listdir(results_path)

        for col in X.columns:

            produced = True
            if "_target".join((col, ".png")) not in result_files:
                produced = False
                break
        self.assertTrue(produced)

        self.tearDown()

    def test_extensive_eda_missing(self):

        X, y = datasets.make_regression(n_samples=100, n_features=5)

        X = pd.DataFrame(X, columns=[f"f_{i}" for i in range(X.shape[1])])
        y = pd.Series(y, name="class")

        ##add some nan values
        X.loc[np.random.randint(0, 100, 20), "f_0"] = np.nan

        results_path = "EDA"
        EDA.extensive_eda(X, y, results_path)
        result_files = os.listdir(results_path)

        for col in X.columns:

            produced = True
            if "_target".join((col, ".png")) not in result_files:
                produced = False
                break
        self.assertTrue(produced)

        X, y = datasets.make_regression(n_samples=100, n_features=5)

        X = pd.DataFrame(X, columns=[f"f_{i}" for i in range(X.shape[1])])
        y = pd.Series(y, name="class")

        ##add some nan values
        X.loc[np.random.randint(0, 100, 20), "f_0"] = np.nan

        results_path = "EDA"
        EDA.extensive_eda(X, y, results_path)
        result_files = os.listdir(results_path)

        for col in X.columns:

            produced = True
            if "_target".join((col, ".png")) not in result_files:
                produced = False
                break
        self.assertTrue(produced)

        self.tearDown()
