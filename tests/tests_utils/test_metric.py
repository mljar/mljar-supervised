import unittest
import numpy as np
import pandas as pd
from numpy.testing import assert_almost_equal

from supervised.utils.metric import Metric
from supervised.utils.metric import UserDefinedEvalMetric


class MetricTest(unittest.TestCase):
    def test_create(self):
        params = {"name": "logloss"}
        m = Metric(params)
        y_true = np.array([0, 0, 1, 1])
        y_predicted = np.array([0, 0, 1, 1])
        score = m(y_true, y_predicted)
        self.assertTrue(score < 0.1)
        y_true = np.array([0, 0, 1, 1])
        y_predicted = np.array([1, 1, 0, 0])
        score = m(y_true, y_predicted)
        self.assertTrue(score > 1.0)

    def test_metric_improvement(self):
        params = {"name": "logloss"}
        m = Metric(params)
        y_true = np.array([0, 0, 1, 1])
        y_predicted = np.array([0, 0, 0, 1])
        score_1 = m(y_true, y_predicted)
        y_true = np.array([0, 0, 1, 1])
        y_predicted = np.array([0, 0, 1, 1])
        score_2 = m(y_true, y_predicted)
        self.assertTrue(m.improvement(score_1, score_2))

    def test_sample_weight(self):
        metrics = ["logloss", "auc", "acc", "rmse", "mse", "mae", "r2", "mape"]
        for m in metrics:
            metric = Metric({"name": m})
            y_true = np.array([0, 0, 1, 1])
            y_predicted = np.array([0, 0, 0, 1])
            sample_weight = np.array([1, 1, 1, 1])

            score_1 = metric(y_true, y_predicted)
            score_2 = metric(y_true, y_predicted, sample_weight)
            assert_almost_equal(score_1, score_2)

    def test_r2_metric(self):
        params = {"name": "r2"}
        m = Metric(params)
        y_true = np.array([0, 0, 1, 1])
        y_predicted = np.array([0, 0, 1, 1])
        score = m(y_true, y_predicted)
        self.assertEqual(score, -1.0)  # negative r2

    def test_mape_metric(self):
        params = {"name": "mape"}
        m = Metric(params)
        y_true = np.array([0, 0, 1, 1])
        y_predicted = np.array([0, 0, 1, 1])
        score = m(y_true, y_predicted)
        self.assertEqual(score, 0.0)

    def test_user_defined_metric(self):
        def custom(x, y, sample_weight=None):
            return np.sum(x + y)

        UserDefinedEvalMetric().set_metric(custom)

        params = {"name": "user_defined_metric"}
        m = Metric(params)

        a = np.array([1, 1, 1])

        score = m(a, a)
        self.assertEqual(score, 6)
