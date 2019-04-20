import unittest
import tempfile
import json
import numpy as np
import pandas as pd

from supervised.models.learner_xgboost import additional


from numpy.testing import assert_almost_equal
from sklearn import datasets

from supervised.iterative_learner_framework import IterativeLearner
from supervised.callbacks.early_stopping import EarlyStopping
from supervised.callbacks.metric_logger import MetricLogger
from supervised.metric import Metric


class MetricLoggerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, cls.y = datasets.make_classification(
            n_samples=200,
            n_features=5,
            n_informative=5,
            n_redundant=0,
            n_classes=2,
            n_clusters_per_class=1,
            n_repeated=0,
            shuffle=False,
            random_state=0,
        )
        cls.data = {
            "train": {
                "X": pd.DataFrame(cls.X, columns=["f0", "f1", "f2", "f3", "f4"]),
                "y": pd.DataFrame(cls.y),
            }
        }

        cls.train_params = {
            "preprocessing": {},
            "validation": {
                "validation_type": "split",
                "train_ratio": 0.5,
                "shuffle": True,
            },
            "learner": {
                "learner_type": "Xgboost",
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "eta": 0.01,
                "silent": 1,
                "max_depth": 1,
                "seed": 1,
            },
        }

    def test_fit_and_predict(self):
        MAX_STEPS = 10
        additional["max_steps"] = MAX_STEPS
        metric_logger = MetricLogger({"metric_names": ["logloss", "auc"]})
        il = IterativeLearner(self.train_params, callbacks=[metric_logger])
        il.train(self.data)
        metric_logs = il.get_metric_logs()
        self.assertEqual(
            len(metric_logs[il.learners[0].uid]["train"]["logloss"]),
            len(metric_logs[il.learners[0].uid]["train"]["auc"]),
        )
        self.assertEqual(
            len(metric_logs[il.learners[0].uid]["train"]["logloss"]),
            len(metric_logs[il.learners[0].uid]["iters"]),
        )
        self.assertEqual(
            len(metric_logs[il.learners[0].uid]["train"]["logloss"]), MAX_STEPS
        )


if __name__ == "__main__":
    unittest.main()
