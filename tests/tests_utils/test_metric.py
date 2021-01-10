import unittest
import numpy as np
import pandas as pd

from supervised.utils.metric import Metric


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

    def test_r2_metric(self):
        params = {"name": "r2"}
        m = Metric(params)
        y_true = np.array([0, 0, 1, 1])
        y_predicted = np.array([0, 0, 1, 1])
        score = m(y_true, y_predicted)
        self.assertEqual(score, -1.0)  # negative r2
