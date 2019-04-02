import unittest
import tempfile
import json
import numpy as np
import pandas as pd

from numpy.testing import assert_almost_equal
from sklearn import datasets

from supervised.models.ensemble import Ensemble
from supervised.metric import Metric


class EnsembleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):

        X = {"model_0": []}
        cls.data = {"train": {"X": X, "y": y}}

    def test_fit_predict(self):
        metric = Metric({"name": "logloss"})

        lgb = LightgbmLearner(self.params)

        loss_prev = None
        for i in range(5):
            lgb.fit(self.data["train"])
            y_predicted = lgb.predict(self.X)
            loss = metric(self.y, y_predicted)
            if loss_prev is not None:
                self.assertTrue(loss + 0.001 < loss_prev)
            loss_prev = loss


if __name__ == "__main__":
    unittest.main()
