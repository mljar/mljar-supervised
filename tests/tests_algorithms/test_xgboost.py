import unittest
import tempfile
import json
import numpy as np
import pandas as pd
import os

from numpy.testing import assert_almost_equal
from sklearn import datasets

from supervised.algorithms.xgboost import XgbAlgorithm, additional
from supervised.utils.metric import Metric

import tempfile

additional["max_rounds"] = 1


class XgboostAlgorithmTest(unittest.TestCase):
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
        for _ in range(3):
            xgb = XgbAlgorithm(params)
            xgb.fit(self.X, self.y)
            y_predicted = xgb.predict(self.X)
            loss = metric(self.y, y_predicted)
            if prev_loss is not None:
                assert_almost_equal(prev_loss, loss)
            prev_loss = loss

    def test_copy(self):
        metric = Metric({"name": "logloss"})
        params = {"objective": "binary:logistic", "eval_metric": "logloss"}
        xgb = XgbAlgorithm(params)
        xgb.fit(self.X, self.y)
        y_predicted = xgb.predict(self.X)
        loss = metric(self.y, y_predicted)

        xgb2 = XgbAlgorithm(params)
        self.assertTrue(xgb2.model is None)  # model is set to None, while initialized
        xgb2 = xgb.copy()
        self.assertEqual(type(xgb), type(xgb2))
        y_predicted = xgb2.predict(self.X)
        loss2 = metric(self.y, y_predicted)
        self.assertEqual(loss, loss2)
        self.assertNotEqual(id(xgb), id(xgb2))

    def test_save_and_load(self):
        metric = Metric({"name": "logloss"})
        params = {"objective": "binary:logistic", "eval_metric": "logloss"}
        xgb = XgbAlgorithm(params)
        xgb.fit(self.X, self.y)
        y_predicted = xgb.predict(self.X)
        loss = metric(self.y, y_predicted)

        filename = os.path.join(tempfile.gettempdir(), os.urandom(12).hex())

        xgb.save(filename)

        xgb2 = XgbAlgorithm(params)
        self.assertTrue(xgb2.model is None)
        xgb2.load(filename)
        # Finished with the file, delete it
        os.remove(filename)

        y_predicted = xgb2.predict(self.X)
        loss2 = metric(self.y, y_predicted)
        assert_almost_equal(loss, loss2)

    def test_restricted_characters_in_feature_name(self):
        df = pd.DataFrame(
            {
                "y": np.random.randint(0, 2, size=100),
                "[test1]": np.random.uniform(0, 1, size=100),
                "test2 < 1": np.random.uniform(0, 1, size=100),
            }
        )

        y = df.iloc[:, 0]
        X = df.iloc[:, 1:]

        metric = Metric({"name": "logloss"})
        params = {"objective": "binary:logistic", "eval_metric": "logloss"}
        xgb = XgbAlgorithm(params)
        xgb.fit(X, y)
        xgb.predict(X)

    def test_get_metric_name(self):
        params = {"objective": "binary:logistic", "eval_metric": "logloss"}
        model = XgbAlgorithm(params)
        self.assertEqual(model.get_metric_name(), "logloss")

        params = {"eval_metric": "rmse"}
        model = XgbAlgorithm(params)
        self.assertEqual(model.get_metric_name(), "rmse")

    def test_is_fitted(self):
        params = {"objective": "binary:logistic", "eval_metric": "logloss"}
        model = XgbAlgorithm(params)
        self.assertFalse(model.is_fitted())
        model.fit(self.X, self.y)
        self.assertTrue(model.is_fitted())
