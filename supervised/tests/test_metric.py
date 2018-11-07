import unittest
import numpy as np
import pandas as pd

from metric import Metric

class MetricTest(unittest.TestCase):

    def test_create(self):
        params = {'metric_name': 'logloss'}
        m = Metric(params)
        y_true = np.array([0,0,1,1])
        y_predicted = np.array([0,0,1,1])
        score = m.score(y_true, y_predicted)
        self.assertTrue(score < 0.1)
        y_true = np.array([0,0,1,1])
        y_predicted = np.array([1,1,0,0])
        score = m.score(y_true, y_predicted)
        self.assertTrue(score > 1.0)
