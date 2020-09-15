import unittest
import numpy as np
import pandas as pd

from supervised.utils.shap import PlotSHAP


class PlotSHAPTest(unittest.TestCase):
    def test_get_sample_data_larger_1k(self):
        """ Get sample when data is larger than 1k """
        X = pd.DataFrame(np.random.uniform(size=(5763, 31)))
        y = pd.Series(np.random.randint(0, 2, size=(5763,)))

        X_, y_ = PlotSHAP.get_sample(X, y)

        self.assertEqual(X_.shape[0], 1000)
        self.assertEqual(y_.shape[0], 1000)

    def test_get_sample_data_smaller_1k(self):
        """ Get sample when data is smaller than 1k """
        SAMPLES = 100
        X = pd.DataFrame(np.random.uniform(size=(SAMPLES, 31)))
        y = pd.Series(np.random.randint(0, 2, size=(SAMPLES,)))

        X_, y_ = PlotSHAP.get_sample(X, y)

        self.assertEqual(X_.shape[0], SAMPLES)
        self.assertEqual(y_.shape[0], SAMPLES)
