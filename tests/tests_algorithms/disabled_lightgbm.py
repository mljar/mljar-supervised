import unittest
import tempfile
import json
import numpy as np
import pandas as pd
import copy
from numpy.testing import assert_almost_equal
from sklearn import datasets

from supervised.algorithms.lightgbm import LightgbmAlgorithm
from supervised.utils.metric import Metric


class LightgbmAlgorithmTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, cls.y = datasets.make_classification(
            n_samples=100,
            n_features=5,
            n_informative=4,
            n_redundant=1,
            n_classes=2,
            n_clusters_per_class=3,
            n_repeated=0,
            shuffle=False,
            random_state=0,
        )
        cls.params = {
            "metric": "binary_logloss",
            "num_leaves": "2",
            "learning_rate": 0.1,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "seed": 1,
        }

    def test_reproduce_fit(self):
        metric = Metric({"name": "logloss"})
        prev_loss = None
        for i in range(3):
            model = LightgbmAlgorithm(self.params)
            model.fit(self.X, self.y)
            y_predicted = model.predict(self.X)
            loss = metric(self.y, y_predicted)
            if prev_loss is not None:
                assert_almost_equal(prev_loss, loss)
            prev_loss = loss

        losses = []
        for i in range(3):
            params = copy.deepcopy(self.params)
            params["seed"] = 1 + i
            model = LightgbmAlgorithm(params)
            model.fit(self.X, self.y)
            y_predicted = model.predict(self.X)
            losses += [metric(self.y, y_predicted)]
        for i in range(1, 3):
            self.assertNotEqual(losses[i - 1], losses[i])

    def test_fit_predict(self):
        metric = Metric({"name": "logloss"})
        lgb = LightgbmAlgorithm(self.params)
        loss_prev = None
        for _ in range(3):
            lgb.fit(self.X, self.y)
            y_predicted = lgb.predict(self.X)
            loss = metric(self.y, y_predicted)
            if loss_prev is not None:
                self.assertTrue(loss + 0.001 < loss_prev)
            loss_prev = loss

    def test_copy(self):
        # train model #1
        metric = Metric({"name": "logloss"})
        lgb = LightgbmAlgorithm(self.params)
        lgb.fit(self.X, self.y)
        y_predicted = lgb.predict(self.X)
        loss = metric(self.y, y_predicted)
        # create model #2
        lgb2 = LightgbmAlgorithm(self.params)
        # model #2 is set to None, while initialized
        self.assertTrue(lgb2.model is None)
        # do a copy and use it for predictions
        lgb2 = lgb.copy()
        self.assertEqual(type(lgb), type(lgb2))
        y_predicted = lgb2.predict(self.X)
        loss2 = metric(self.y, y_predicted)
        self.assertEqual(loss, loss2)

    def test_save_and_load(self):
        metric = Metric({"name": "logloss"})
        lgb = LightgbmAlgorithm(self.params)
        lgb.fit(self.X, self.y)
        y_predicted = lgb.predict(self.X)
        loss = metric(self.y, y_predicted)

        json_desc = lgb.save()
        lgb2 = LightgbmAlgorithm({})
        self.assertTrue(lgb.uid != lgb2.uid)
        self.assertTrue(lgb2.model is None)
        lgb2.load(json_desc)
        self.assertTrue(lgb.uid == lgb2.uid)

        y_predicted = lgb2.predict(self.X)
        loss2 = metric(self.y, y_predicted)
        assert_almost_equal(loss, loss2)


if __name__ == "__main__":
    unittest.main()
