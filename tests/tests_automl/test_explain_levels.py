import os
import unittest
import tempfile
import json
import numpy as np
import pandas as pd
import shutil
from sklearn import datasets

from supervised import AutoML

from supervised.algorithms.random_forest import additional

additional["max_steps"] = 1
additional["trees_in_step"] = 1

from supervised.algorithms.xgboost import additional

additional["max_rounds"] = 1


class AutoMLExplainLevelsTest(unittest.TestCase):

    automl_dir = "automl_1"

    def tearDown(self):
        shutil.rmtree(self.automl_dir, ignore_errors=True)

    def test_explain_default(self):
        a = AutoML(
            results_path=self.automl_dir,
            total_time_limit=1,
            algorithms=["Random Forest"],
            train_ensemble=False,
            validation={
                "validation_type": "kfold",
                "k_folds": 2,
                "shuffle": True,
                "stratify": True,
            },
        )
        a.set_advanced(start_random_models=1)

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
        X = pd.DataFrame(X, columns=[f"f_{i}" for i in range(X.shape[1])])

        a.fit(X, y)

        result_files = os.listdir(
            os.path.join(self.automl_dir, "1_Default_RandomForest")
        )

        # There should be files with:
        # - permutation importance
        # - shap importance
        # - shap dependence
        # - shap decisions

        # Check permutation importance
        produced = False
        for f in result_files:
            if "importance.csv" in f and "shap" not in f:
                produced = True
                break
        self.assertTrue(produced)
        # Check shap importance
        produced = False
        for f in result_files:
            if "importance.csv" in f and "shap" in f:
                produced = True
                break
        self.assertTrue(produced)
        # Check shap dependence
        produced = False
        for f in result_files:
            if "dependence.png" in f:
                produced = True
                break
        self.assertTrue(produced)
        # Check shap decisions
        produced = False
        for f in result_files:
            if "decisions.png" in f:
                produced = True
                break
        self.assertTrue(produced)

    def test_no_explain_linear(self):
        a = AutoML(
            results_path=self.automl_dir,
            total_time_limit=1,
            algorithms=["Linear"],
            train_ensemble=False,
            validation={
                "validation_type": "kfold",
                "k_folds": 2,
                "shuffle": True,
                "stratify": True,
            },
            explain_level=0,
        )
        a.set_advanced(start_random_models=1)

        X, y = datasets.make_regression(
            n_samples=100, n_features=5, n_informative=4, shuffle=False, random_state=0
        )
        X = pd.DataFrame(X, columns=[f"f_{i}" for i in range(X.shape[1])])

        a.fit(X, y)

        result_files = os.listdir(os.path.join(self.automl_dir, "1_Linear"))

        # There should be no files with:
        # - permutation importance
        # - shap importance
        # - shap dependence
        # - shap decisions

        # Check permutation importance
        produced = False
        for f in result_files:
            if "importance.csv" in f and "shap" not in f:
                produced = True
                break
        self.assertFalse(produced)
        # Check shap importance
        produced = False
        for f in result_files:
            if "importance.csv" in f and "shap" in f:
                produced = True
                break
        self.assertFalse(produced)
        # Check shap dependence
        produced = False
        for f in result_files:
            if "dependence.png" in f:
                produced = True
                break
        self.assertFalse(produced)
        # Check shap decisions
        produced = False
        for f in result_files:
            if "decisions.png" in f:
                produced = True
                break
        self.assertFalse(produced)
        # Check coefficients
        produced = False
        for f in result_files:
            if "coefs.csv" in f:
                produced = True
                break
        self.assertFalse(produced)

    def test_explain_just_permutation_importance(self):
        a = AutoML(
            results_path=self.automl_dir,
            total_time_limit=1,
            algorithms=["Xgboost"],
            train_ensemble=False,
            validation={
                "validation_type": "kfold",
                "k_folds": 2,
                "shuffle": True,
                "stratify": True,
            },
            explain_level=1,
        )
        a.set_advanced(start_random_models=1)

        X, y = datasets.make_regression(
            n_samples=100, n_features=5, n_informative=4, shuffle=False, random_state=0
        )
        X = pd.DataFrame(X, columns=[f"f_{i}" for i in range(X.shape[1])])

        a.fit(X, y)

        result_files = os.listdir(os.path.join(self.automl_dir, "1_Default_Xgboost"))

        # There should be no files with:
        # - permutation importance
        # - shap importance
        # - shap dependence
        # - shap decisions

        # Check permutation importance
        produced = False
        for f in result_files:
            if "importance.csv" in f and "shap" not in f:
                produced = True
                break
        self.assertTrue(produced)
        # Check shap importance
        produced = False
        for f in result_files:
            if "importance.csv" in f and "shap" in f:
                produced = True
                break
        self.assertFalse(produced)
        # Check shap dependence
        produced = False
        for f in result_files:
            if "dependence.png" in f:
                produced = True
                break
        self.assertFalse(produced)
        # Check shap decisions
        produced = False
        for f in result_files:
            if "decisions.png" in f:
                produced = True
                break
        self.assertFalse(produced)

    def test_build_decision_tree(self):
        a = AutoML(
            results_path=self.automl_dir,
            total_time_limit=1,
            algorithms=["Decision Tree"],
            train_ensemble=False,
            validation={
                "validation_type": "kfold",
                "k_folds": 2,
                "shuffle": True,
                "stratify": True,
            },
            explain_level=2,
        )
        a.set_advanced(start_random_models=1)

        X, y = datasets.make_regression(
            n_samples=100, n_features=5, n_informative=4, shuffle=False, random_state=0
        )
        X = pd.DataFrame(X, columns=[f"f_{i}" for i in range(X.shape[1])])

        a.fit(X, y)

        result_files = os.listdir(os.path.join(self.automl_dir, "1_DecisionTree"))

        # There should be files with:
        # - decision tree visualization
        # - permutation importance
        # - shap importance
        # - shap dependence
        # - shap decisions

        # Check Decision Tree visualization
        produced = False
        for f in result_files:
            if "tree.svg" in f:
                produced = True
                break
        self.assertTrue(produced)
        # Check permutation importance
        produced = False
        for f in result_files:
            if "importance.csv" in f and "shap" not in f:
                produced = True
                break
        self.assertTrue(produced)
        # Check shap importance
        produced = False
        for f in result_files:
            if "importance.csv" in f and "shap" in f:
                produced = True
                break
        self.assertTrue(produced)
        # Check shap dependence
        produced = False
        for f in result_files:
            if "dependence.png" in f:
                produced = True
                break
        self.assertTrue(produced)
        # Check shap decisions
        produced = False
        for f in result_files:
            if "decisions.png" in f:
                produced = True
                break
        self.assertTrue(produced)
