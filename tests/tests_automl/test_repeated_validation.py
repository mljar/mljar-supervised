import os
import unittest
import tempfile
import json
import numpy as np
import pandas as pd
import shutil
from sklearn import datasets

from supervised import AutoML
from supervised.utils.common import construct_learner_name

from supervised.algorithms.random_forest import additional

additional["max_steps"] = 1
additional["trees_in_step"] = 1

from supervised.algorithms.xgboost import additional

additional["max_rounds"] = 1


class AutoMLRepeatedValidationTest(unittest.TestCase):

    automl_dir = "automl_1"

    def tearDown(self):
        shutil.rmtree(self.automl_dir, ignore_errors=True)

    def test_repeated_kfold(self):
        REPEATS = 3
        FOLDS = 2

        a = AutoML(
            results_path=self.automl_dir,
            total_time_limit=10,
            algorithms=["Random Forest"],
            train_ensemble=False,
            validation_strategy={
                "validation_type": "kfold",
                "k_folds": FOLDS,
                "repeats": REPEATS,
                "shuffle": True,
                "stratify": True,
            },
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
        X = pd.DataFrame(X, columns=[f"f_{i}" for i in range(X.shape[1])])

        a.fit(X, y)

        result_files = os.listdir(
            os.path.join(self.automl_dir, "1_Default_RandomForest")
        )

        cnt = 0
        for repeat in range(REPEATS):
            for fold in range(FOLDS):
                learner_name = construct_learner_name(fold, repeat, REPEATS)
                self.assertTrue(f"{learner_name}.random_forest" in result_files)
                self.assertTrue(f"{learner_name}_training.log" in result_files)
                cnt += 1
        self.assertTrue(cnt, 6)

    def test_repeated_split(self):
        REPEATS = 3
        FOLDS = 1

        a = AutoML(
            results_path=self.automl_dir,
            total_time_limit=10,
            algorithms=["Random Forest"],
            train_ensemble=False,
            validation_strategy={
                "validation_type": "split",
                "repeats": REPEATS,
                "shuffle": True,
                "stratify": True,
            },
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
        X = pd.DataFrame(X, columns=[f"f_{i}" for i in range(X.shape[1])])

        a.fit(X, y)

        result_files = os.listdir(
            os.path.join(self.automl_dir, "1_Default_RandomForest")
        )
        cnt = 0
        for repeat in range(REPEATS):
            for fold in range(FOLDS):
                learner_name = construct_learner_name(fold, repeat, REPEATS)
                self.assertTrue(f"{learner_name}.random_forest" in result_files)
                self.assertTrue(f"{learner_name}_training.log" in result_files)
                cnt += 1
        self.assertTrue(cnt, 3)
