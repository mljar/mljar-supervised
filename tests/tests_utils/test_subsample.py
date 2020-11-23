import os
import unittest
import numpy as np
import pandas as pd
import tempfile

from supervised.utils.subsample import subsample
from supervised.algorithms.registry import REGRESSION


class SubsampleTest(unittest.TestCase):
    def test_subsample_regression_10k(self):

        rows = 10000
        cols = 51
        X = np.random.rand(rows, cols)
        X = pd.DataFrame(X, columns=[f"f{i}" for i in range(cols)])
        y = pd.Series(np.random.rand(rows), name="target")

        X_train, X_test, y_train, y_test = subsample(
            X, y, train_size=1000, ml_task=REGRESSION
        )

        self.assertTrue(X_train.shape[0], 1000)
        self.assertTrue(X_test.shape[0], 9000)
        self.assertTrue(y_train.shape[0], 1000)
        self.assertTrue(y_test.shape[0], 9000)
