import shutil
import unittest

import numpy as np
import pandas as pd

from supervised import AutoML
from supervised.algorithms.random_forest import additional
from supervised.algorithms.registry import MULTICLASS_CLASSIFICATION

additional["max_steps"] = 1
additional["trees_in_step"] = 1

from supervised.algorithms.xgboost import additional

additional["max_rounds"] = 1


class AutoMLHandleImbalanceTest(unittest.TestCase):
    automl_dir = "AutoMLHandleImbalanceTest"

    def tearDown(self):
        shutil.rmtree(self.automl_dir, ignore_errors=True)

    def test_handle_drastic_imbalance(self):
        a = AutoML(
            results_path=self.automl_dir,
            total_time_limit=10,
            algorithms=["Random Forest"],
            train_ensemble=False,
            validation_strategy={
                "validation_type": "kfold",
                "k_folds": 10,
                "shuffle": True,
                "stratify": True,
            },
            start_random_models=1,
        )

        rows = 100
        X = pd.DataFrame(
            {
                "f1": np.random.rand(rows),
                "f2": np.random.rand(rows),
                "f3": np.random.rand(rows),
            }
        )
        y = np.ones(rows)

        y[:8] = 0
        y[10:12] = 2
        y = pd.Series(np.array(y), name="target")
        a._ml_task = MULTICLASS_CLASSIFICATION
        a._handle_drastic_imbalance(X, y)

        self.assertEqual(X.shape[0], 130)
        self.assertEqual(X.shape[1], 3)
        self.assertEqual(y.shape[0], 130)

    def test_handle_drastic_imbalance_sample_weight(self):
        a = AutoML(
            results_path=self.automl_dir,
            total_time_limit=10,
            algorithms=["Random Forest"],
            train_ensemble=False,
            validation_strategy={
                "validation_type": "kfold",
                "k_folds": 10,
                "shuffle": True,
                "stratify": True,
            },
            start_random_models=1,
        )

        rows = 100
        X = pd.DataFrame(
            {
                "f1": np.random.rand(rows),
                "f2": np.random.rand(rows),
                "f3": np.random.rand(rows),
            }
        )
        y = np.ones(rows)
        sample_weight = pd.Series(np.array(range(rows)), name="sample_weight")

        y[:1] = 0
        y[10:11] = 2

        y = pd.Series(np.array(y), name="target")
        a._ml_task = MULTICLASS_CLASSIFICATION
        a._handle_drastic_imbalance(X, y, sample_weight)

        self.assertEqual(X.shape[0], 138)
        self.assertEqual(X.shape[1], 3)
        self.assertEqual(y.shape[0], 138)

        self.assertEqual(np.sum(sample_weight[100:119]), 0)
        self.assertEqual(np.sum(sample_weight[119:138]), 19 * 10)

    def test_imbalance_dont_change_data_after_fit(self):
        a = AutoML(
            results_path=self.automl_dir,
            total_time_limit=5,
            train_ensemble=False,
            validation_strategy={
                "validation_type": "kfold",
                "k_folds": 10,
                "shuffle": True,
                "stratify": True,
            },
            start_random_models=1,
            explain_level=0,
        )

        rows = 100
        X = pd.DataFrame(
            {
                "f1": np.random.rand(rows),
                "f2": np.random.rand(rows),
                "f3": np.random.rand(rows),
            }
        )
        y = np.ones(rows)

        y[:8] = 0
        y[10:12] = 2
        sample_weight = np.ones(rows)

        a.fit(X, y, sample_weight=sample_weight)

        # original data **without** inserted samples to handle imbalance
        self.assertEqual(X.shape[0], rows)
        self.assertEqual(y.shape[0], rows)
        self.assertEqual(sample_weight.shape[0], rows)
