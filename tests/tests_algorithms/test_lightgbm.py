import unittest
import tempfile
import json
import copy
import os

import numpy as np
import pandas as pd
from numpy.testing import assert_almost_equal
from sklearn import datasets

from supervised.algorithms.lightgbm import LightgbmAlgorithm, additional
from supervised.utils.metric import Metric

additional["max_rounds"] = 1


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

        filename = os.path.join(tempfile.gettempdir(), os.urandom(12).hex())
        lgb.save(filename)
        lgb2 = LightgbmAlgorithm({})
        self.assertTrue(lgb.uid != lgb2.uid)
        self.assertTrue(lgb2.model is None)
        lgb2.load(filename)
        # Finished with the file, delete it
        os.remove(filename)

        y_predicted = lgb2.predict(self.X)
        loss2 = metric(self.y, y_predicted)
        assert_almost_equal(loss, loss2)

    def test_get_metric_name(self):
        model = LightgbmAlgorithm(self.params)
        self.assertEqual(model.get_metric_name(), "logloss")

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
        lgb = LightgbmAlgorithm(self.params)
        lgb.fit(X, y)
        lgb.predict(X)


if __name__ == "__main__":
    unittest.main()
