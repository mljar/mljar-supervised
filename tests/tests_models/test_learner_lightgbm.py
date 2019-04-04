import unittest
import tempfile
import json
import numpy as np
import pandas as pd

from numpy.testing import assert_almost_equal
from sklearn import datasets

from supervised.models.learner_lightgbm import LightgbmLearner
from supervised.metric import Metric


class LightgbmLearnerTest(unittest.TestCase):
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
        }

    def test_fit_predict(self):
        metric = Metric({"name": "logloss"})

        lgb = LightgbmLearner(self.params)

        loss_prev = None
        for i in range(5):
            lgb.fit(self.X, self.y)
            y_predicted = lgb.predict(self.X)
            loss = metric(self.y, y_predicted)
            if loss_prev is not None:
                self.assertTrue(loss + 0.001 < loss_prev)
            loss_prev = loss

    def test_copy(self):
        # train model #1
        metric = Metric({"name": "logloss"})
        lgb = LightgbmLearner(self.params)
        lgb.fit(self.X, self.y)
        y_predicted = lgb.predict(self.X)
        loss = metric(self.y, y_predicted)
        # create model #2
        lgb2 = LightgbmLearner(self.params)
        # model #2 is set to None, while initialized
        self.assertTrue(lgb2.model is None)
        # do a copy and use it for predictions
        lgb2 = lgb.copy()
        self.assertEqual(type(lgb), type(lgb2))
        y_predicted = lgb2.predict(self.X)
        loss2 = metric(self.y, y_predicted)
        self.assertEqual(loss, loss2)
        # fit model #1, there should be improvement in loss
        lgb.fit(self.X, self.y)
        y_predicted = lgb.predict(self.X)
        loss3 = metric(self.y, y_predicted)
        self.assertTrue(loss3 < loss)
        # the loss of model #2 should not change
        y_predicted = lgb2.predict(self.X)
        loss4 = metric(self.y, y_predicted)
        assert_almost_equal(loss2, loss4)

    def test_save_and_load(self):
        metric = Metric({"name": "logloss"})
        lgb = LightgbmLearner(self.params)
        lgb.fit(self.X, self.y)
        y_predicted = lgb.predict(self.X)
        loss = metric(self.y, y_predicted)

        json_desc = lgb.save()
        lgb2 = LightgbmLearner({})
        self.assertTrue(lgb.uid != lgb2.uid)
        self.assertTrue(lgb2.model is None)
        lgb2.load(json_desc)
        self.assertTrue(lgb.uid == lgb2.uid)

        y_predicted = lgb2.predict(self.X)
        loss2 = metric(self.y, y_predicted)
        assert_almost_equal(loss, loss2)


if __name__ == "__main__":
    unittest.main()
