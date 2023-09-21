import os
import unittest
import tempfile
import json
import numpy as np
import pandas as pd
import shutil
from sklearn import datasets
from sklearn.model_selection import train_test_split
from numpy.testing import assert_almost_equal
from supervised import AutoML


class AutoMLPredictionAfterLoadTest(unittest.TestCase):
    automl_dir = "automl_testing"

    def tearDown(self):
        shutil.rmtree(self.automl_dir, ignore_errors=True)

    def test_integration(self):
        a = AutoML(
            results_path=self.automl_dir,
            mode="Compete",
            algorithms=["Baseline", "CatBoost", "LightGBM", "Xgboost"],
            stack_models=True,
            total_time_limit=60,
            validation_strategy={
                "validation_type": "kfold",
                "k_folds": 3,
                "shuffle": True,
                "stratify": True,
                "random_seed": 123,
            },
        )

        X, y = datasets.make_classification(
            n_samples=1000,
            n_features=30,
            n_informative=29,
            n_redundant=1,
            n_classes=8,
            n_clusters_per_class=3,
            n_repeated=0,
            shuffle=False,
            random_state=0,
        )
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

        a.fit(X_train, y_train)
        p = a.predict_all(X_test)

        a2 = AutoML(results_path=self.automl_dir)
        p2 = a2.predict_all(X_test)

        assert_almost_equal(p["prediction_0"].iloc[0], p2["prediction_0"].iloc[0])
        assert_almost_equal(p["prediction_7"].iloc[0], p2["prediction_7"].iloc[0])
