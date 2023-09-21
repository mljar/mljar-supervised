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


class AutoMLGoldenFeaturesTest(unittest.TestCase):
    automl_dir = "automl_tests"
    rows = 50

    def tearDown(self):
        shutil.rmtree(self.automl_dir, ignore_errors=True)

    def test_no_golden_features(self):
        N_COLS = 10
        X, y = datasets.make_classification(
            n_samples=100,
            n_features=N_COLS,
            n_informative=6,
            n_redundant=1,
            n_classes=2,
            n_clusters_per_class=3,
            n_repeated=0,
            shuffle=False,
            random_state=0,
        )

        X = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])

        automl = AutoML(
            results_path=self.automl_dir,
            total_time_limit=50,
            algorithms=["Xgboost"],
            train_ensemble=False,
            golden_features=False,
            explain_level=0,
            start_random_models=1,
        )
        automl.fit(X, y)

        self.assertEqual(len(automl._models), 1)

    def test_golden_features(self):
        N_COLS = 10
        X, y = datasets.make_classification(
            n_samples=100,
            n_features=N_COLS,
            n_informative=6,
            n_redundant=1,
            n_classes=2,
            n_clusters_per_class=3,
            n_repeated=0,
            shuffle=False,
            random_state=0,
        )

        X = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])

        automl = AutoML(
            results_path=self.automl_dir,
            total_time_limit=50,
            algorithms=["Xgboost"],
            train_ensemble=False,
            golden_features=True,
            explain_level=0,
            start_random_models=1,
        )
        automl.fit(X, y)

        self.assertEqual(len(automl._models), 2)

        # there should be 10 golden features
        with open(os.path.join(self.automl_dir, "golden_features.json")) as fin:
            d = json.loads(fin.read())
            self.assertEqual(len(d["new_features"]), 10)

    def test_golden_features_count(self):
        N_COLS = 10
        X, y = datasets.make_classification(
            n_samples=100,
            n_features=N_COLS,
            n_informative=6,
            n_redundant=1,
            n_classes=2,
            n_clusters_per_class=3,
            n_repeated=0,
            shuffle=False,
            random_state=0,
        )

        X = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])

        automl = AutoML(
            results_path=self.automl_dir,
            total_time_limit=50,
            algorithms=["Xgboost"],
            train_ensemble=False,
            golden_features=50,
            explain_level=0,
            start_random_models=1,
        )
        automl.fit(X, y)

        self.assertEqual(len(automl._models), 2)

        # there should be 50 golden features
        with open(os.path.join(self.automl_dir, "golden_features.json")) as fin:
            d = json.loads(fin.read())
            self.assertEqual(len(d["new_features"]), 50)
