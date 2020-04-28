import unittest
import tempfile
import json
import numpy as np
import pandas as pd

from numpy.testing import assert_almost_equal
from sklearn import datasets

from supervised.algorithms.linear import (LinearAlgorithm, LinearRegressorAlgorithm)
from supervised.utils.metric import Metric

import tempfile


class LinearRegressorAlgorithmTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, cls.y = datasets.make_regression(
            n_samples=100,
            n_features=5,
            n_informative=4,
            shuffle=False,
            random_state=0,
        )

    def test_reproduce_fit(self):
        metric = Metric({"name": "mse"})
        params = {"seed": 1, "ml_task": "regression"}
        prev_loss = None
        for _ in range(3):
            model = LinearRegressorAlgorithm(params)
            model.fit(self.X, self.y)
            y_predicted = model.predict(self.X)
            loss = metric(self.y, y_predicted)
            if prev_loss is not None:
                assert_almost_equal(prev_loss, loss)
            prev_loss = loss

class LinearAlgorithmTest(unittest.TestCase):
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
        params = {"seed": 1, "ml_task": "binary_classification"}
        prev_loss = None
        for _ in range(3):
            model = LinearAlgorithm(params)
            model.fit(self.X, self.y)
            y_predicted = model.predict(self.X)
            loss = metric(self.y, y_predicted)
            if prev_loss is not None:
                assert_almost_equal(prev_loss, loss)
            prev_loss = loss

    def test_fit_predict(self):
        metric = Metric({"name": "logloss"})
        params = {"ml_task": "binary_classification"}
        la = LinearAlgorithm(params)

        la.fit(self.X, self.y)
        y_predicted = la.predict(self.X)
        self.assertTrue(metric(self.y, y_predicted) < 0.6)

    def test_copy(self):
        metric = Metric({"name": "logloss"})
        model = LinearAlgorithm({"ml_task": "binary_classification"})
        model.fit(self.X, self.y)
        y_predicted = model.predict(self.X)
        loss = metric(self.y, y_predicted)

        model2 = LinearAlgorithm({})
        model2 = model.copy()
        self.assertEqual(type(model), type(model2))
        y_predicted = model2.predict(self.X)
        loss2 = metric(self.y, y_predicted)
        assert_almost_equal(loss, loss2)

    def test_save_and_load(self):
        metric = Metric({"name": "logloss"})
        model = LinearAlgorithm({"ml_task": "binary_classification"})
        model.fit(self.X, self.y)
        y_predicted = model.predict(self.X)
        loss = metric(self.y, y_predicted)

        with tempfile.NamedTemporaryFile() as tmp:

            model.save(tmp.name)
            model2 = LinearAlgorithm({"ml_task": "binary_classification"})
            model2.load(tmp.name)

            y_predicted = model2.predict(self.X)
            loss2 = metric(self.y, y_predicted)
            assert_almost_equal(loss, loss2)
