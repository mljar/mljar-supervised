import unittest
import tempfile
import json
import numpy as np
import pandas as pd

from numpy.testing import assert_almost_equal
from sklearn import datasets

from supervised.algorithms.compute_additional_metrics import ComputeAdditionalMetrics
from supervised.utils.metric import Metric
from supervised.algorithms.registry import BINARY_CLASSIFICATION


class ComputeAdditionalMetricsTest(unittest.TestCase):
    def test_compute(self):
        target = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        pred = np.array([0.1, 0.8, 0.1, 0.1, 0.8, 0.1, 0.8, 0.8])
        details, max_metrics, conf = ComputeAdditionalMetrics.compute(
            target, pred, BINARY_CLASSIFICATION
        )
        self.assertEqual(conf.iloc[0, 0], 3)
        self.assertEqual(conf.iloc[1, 1], 3)
        self.assertTrue(details is not None)
        self.assertTrue(max_metrics is not None)

    def test_compute_f1(self):
        target = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        pred = np.array([0.01, 0.2, 0.1, 0.1, 0.8, 0.8, 0.8, 0.8])
        details, max_metrics, conf = ComputeAdditionalMetrics.compute(
            target, pred, BINARY_CLASSIFICATION
        )
        self.assertEqual(max_metrics["f1"]["score"], 1)
        self.assertTrue(details is not None)
        self.assertTrue(conf is not None)

if __name__ == "__main__":
    unittest.main()
