import unittest
import tempfile
import json
import numpy as np
import pandas as pd

from supervised.algorithms.xgboost import additional


from numpy.testing import assert_almost_equal
from sklearn import datasets

from supervised.model_framework import ModelFramework
from supervised.callbacks.early_stopping import EarlyStopping
from supervised.callbacks.metric_logger import MetricLogger
from supervised.callbacks.max_iters_constraint import MaxItersConstraint
from supervised.utils.metric import Metric


class MaxItersConstraintTest(unittest.TestCase):
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

        cls.kfolds = 3
        cls.train_params = {
            "preprocessing": {},
            "validation": {"validation_type": "kfold", "kfold": cls.kfolds},
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
        MAX_STEPS = 100
        additional["max_steps"] = MAX_STEPS
        iters_cnt = 5
        max_iters = MaxItersConstraint({"max_iters": iters_cnt})
        metric_logger = MetricLogger({"metric_names": ["logloss"]})
        il = ModelFramework(self.train_params, callbacks=[max_iters, metric_logger])
        il.train(self.data)
        metric_logs = il.get_metric_logs()
        for k in range(self.kfolds):
            self.assertEqual(
                len(metric_logs[il.learners[k].uid]["train"]["logloss"]), iters_cnt
            )
            self.assertNotEqual(
                len(metric_logs[il.learners[k].uid]["train"]["logloss"]), MAX_STEPS
            )


if __name__ == "__main__":
    unittest.main()
