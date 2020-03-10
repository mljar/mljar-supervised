import unittest
import tempfile
import json
import copy
import numpy as np
import pandas as pd

from numpy.testing import assert_almost_equal
from sklearn import datasets

from supervised.algorithms.catboost import CatBoostAlgorithm
from supervised.utils.metric import Metric


class CatBoostAlgorithmTest(unittest.TestCase):
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
            "learning_rate": 0.1,
            "depth": 2,
            "rsm": 0.5,
            "random_strength": 1,
            "bagging_temperature": 0.7,
            "l2_leaf_reg": 1,
            "seed": 1,
        }

    def test_reproduce_fit(self):
        metric = Metric({"name": "logloss"})
        prev_loss = None
        for _ in range(3):
            model = CatBoostAlgorithm(self.params)
            model.fit(self.X, self.y)
            y_predicted = model.predict(self.X)
            loss = metric(self.y, y_predicted)
            if prev_loss is not None:
                assert_almost_equal(prev_loss, loss)
            prev_loss = loss

    def test_fit_predict(self):
        metric = Metric({"name": "logloss"})
        cat = CatBoostAlgorithm(self.params)
        loss_prev = None
        for _ in range(5):
            cat.fit(self.X, self.y)
            y_predicted = cat.predict(self.X)
            loss = metric(self.y, y_predicted)
            if loss_prev is not None:
                self.assertTrue(loss + 0.001 < loss_prev)
            loss_prev = loss

    def test_copy(self):
        # train model #1
        metric = Metric({"name": "logloss"})
        cat = CatBoostAlgorithm(self.params)
        cat.fit(self.X, self.y)
        y_predicted = cat.predict(self.X)
        loss = metric(self.y, y_predicted)
        # create model #2
        cat2 = CatBoostAlgorithm(self.params)
        # model #2 is initialized in constructor
        self.assertTrue(cat2.model is not None)
        # do a copy and use it for predictions
        cat2 = cat.copy()
        self.assertEqual(type(cat), type(cat2))
        y_predicted = cat2.predict(self.X)
        loss2 = metric(self.y, y_predicted)
        self.assertEqual(loss, loss2)
        # fit model #1, there should be improvement in loss
        cat.fit(self.X, self.y)
        y_predicted = cat.predict(self.X)
        loss3 = metric(self.y, y_predicted)
        self.assertTrue(loss3 < loss)
        # the loss of model #2 should not change
        y_predicted = cat2.predict(self.X)
        loss4 = metric(self.y, y_predicted)
        assert_almost_equal(loss2, loss4)

    def test_save_and_load(self):
        metric = Metric({"name": "logloss"})
        cat = CatBoostAlgorithm(self.params)
        cat.fit(self.X, self.y)
        y_predicted = cat.predict(self.X)
        loss = metric(self.y, y_predicted)

        json_desc = cat.save()
        cat2 = CatBoostAlgorithm({})
        self.assertTrue(cat.uid != cat2.uid)
        self.assertTrue(cat2.model is not None)
        cat2.load(json_desc)
        self.assertTrue(cat.uid == cat2.uid)

        y_predicted = cat2.predict(self.X)
        loss2 = metric(self.y, y_predicted)
        assert_almost_equal(loss, loss2)


if __name__ == "__main__":
    unittest.main()
