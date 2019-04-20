import unittest
import tempfile
import json
import numpy as np
import pandas as pd

from numpy.testing import assert_almost_equal
from sklearn import datasets

from supervised.models.learner_xgboost import XgbLearner
from supervised.metric import Metric


class XgboostLearnerTest(unittest.TestCase):
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

    def test_reproduce_fit(self):
        metric = Metric({"name": "logloss"})
        params = {"objective": "binary:logistic", "eval_metric": "logloss", "seed": 1}
        prev_loss = None
        for i in range(3):
            xgb = XgbLearner(params)
            xgb.fit(self.X, self.y)
            y_predicted = xgb.predict(self.X)
            loss = metric(self.y, y_predicted)
            if prev_loss is not None:
                assert_almost_equal(prev_loss, loss)
            prev_loss = loss

    def test_fit_predict(self):
        metric = Metric({"name": "logloss"})
        params = {"objective": "binary:logistic", "eval_metric": "logloss"}
        xgb = XgbLearner(params)

        loss_prev = None
        for i in range(5):
            xgb.fit(self.X, self.y)
            y_predicted = xgb.predict(self.X)
            loss = metric(self.y, y_predicted)
            if loss_prev is not None:
                self.assertTrue(loss + 0.001 < loss_prev)
            loss_prev = loss

    def test_copy(self):
        metric = Metric({"name": "logloss"})
        params = {"objective": "binary:logistic", "eval_metric": "logloss"}
        xgb = XgbLearner(params)
        xgb.fit(self.X, self.y)
        y_predicted = xgb.predict(self.X)
        loss = metric(self.y, y_predicted)

        xgb2 = XgbLearner(params)
        self.assertTrue(xgb2.model is None)  # model is set to None, while initialized
        xgb2 = xgb.copy()
        self.assertEqual(type(xgb), type(xgb2))
        y_predicted = xgb2.predict(self.X)
        loss2 = metric(self.y, y_predicted)
        self.assertEqual(loss, loss2)

        xgb.fit(self.X, self.y)
        y_predicted = xgb.predict(self.X)
        loss3 = metric(self.y, y_predicted)
        self.assertTrue(loss3 < loss)

        y_predicted = xgb2.predict(self.X)
        loss4 = metric(self.y, y_predicted)
        assert_almost_equal(loss2, loss4)

    def test_save_and_load(self):
        metric = Metric({"name": "logloss"})
        params = {"objective": "binary:logistic", "eval_metric": "logloss"}
        xgb = XgbLearner(params)
        xgb.fit(self.X, self.y)
        y_predicted = xgb.predict(self.X)
        loss = metric(self.y, y_predicted)

        json_desc = xgb.save()
        xgb2 = XgbLearner(params)
        self.assertTrue(xgb.uid != xgb2.uid)
        self.assertTrue(xgb2.model is None)
        xgb2.load(json_desc)
        self.assertTrue(xgb.uid == xgb2.uid)

        y_predicted = xgb2.predict(self.X)
        loss2 = metric(self.y, y_predicted)
        assert_almost_equal(loss, loss2)


if __name__ == "__main__":
    unittest.main()
