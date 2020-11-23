import os
import unittest
import tempfile
import json
import numpy as np
import pandas as pd
import shutil
from sklearn import datasets

from supervised import AutoML
from supervised.algorithms.registry import MULTICLASS_CLASSIFICATION
from supervised.algorithms.random_forest import additional

additional["max_steps"] = 1
additional["trees_in_step"] = 1

from supervised.algorithms.xgboost import additional

additional["max_rounds"] = 1


class AutoMLHandleImbalanceTest(unittest.TestCase):

    automl_dir = "automl_1"

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
