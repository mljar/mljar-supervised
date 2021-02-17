import unittest
import tempfile
import json
import numpy as np
import pandas as pd
import os

from numpy.testing import assert_almost_equal
from sklearn import datasets

from supervised.algorithms.extra_trees import (
    ExtraTreesAlgorithm,
    ExtraTreesRegressorAlgorithm,
    additional,
    regression_additional,
)
from supervised.utils.metric import Metric

import tempfile

additional["trees_in_step"] = 1
regression_additional["trees_in_step"] = 1
additional["max_steps"] = 1
regression_additional["max_steps"] = 1


class ExtraTreesRegressorAlgorithmTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, cls.y = datasets.make_regression(
            n_samples=100, n_features=5, n_informative=4, shuffle=False, random_state=0
        )

    def test_reproduce_fit(self):
        metric = Metric({"name": "mse"})
        params = {"trees_in_step": 1, "seed": 1, "ml_task": "regression"}
        prev_loss = None
        for _ in range(3):
            model = ExtraTreesRegressorAlgorithm(params)
            model.fit(self.X, self.y)
            y_predicted = model.predict(self.X)
            loss = metric(self.y, y_predicted)
            if prev_loss is not None:
                assert_almost_equal(prev_loss, loss)
            prev_loss = loss


class ExtraTreesAlgorithmTest(unittest.TestCase):
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
        params = {"trees_in_step": 1, "seed": 1, "ml_task": "binary_classification"}
        prev_loss = None
        for _ in range(3):
            model = ExtraTreesAlgorithm(params)
            model.fit(self.X, self.y)
            y_predicted = model.predict(self.X)
            loss = metric(self.y, y_predicted)
            if prev_loss is not None:
                assert_almost_equal(prev_loss, loss)
            prev_loss = loss

    def test_fit_predict(self):
        metric = Metric({"name": "logloss"})
        params = {"trees_in_step": 50, "ml_task": "binary_classification"}
        rf = ExtraTreesAlgorithm(params)

        rf.fit(self.X, self.y)
        y_predicted = rf.predict(self.X)
        self.assertTrue(metric(self.y, y_predicted) < 0.6)

    def test_copy(self):
        metric = Metric({"name": "logloss"})
        rf = ExtraTreesAlgorithm({"ml_task": "binary_classification"})
        rf.fit(self.X, self.y)
        y_predicted = rf.predict(self.X)
        loss = metric(self.y, y_predicted)

        rf2 = ExtraTreesAlgorithm({"ml_task": "binary_classification"})
        rf2 = rf.copy()
        self.assertEqual(type(rf), type(rf2))
        y_predicted = rf2.predict(self.X)
        loss2 = metric(self.y, y_predicted)
        assert_almost_equal(loss, loss2)

    def test_save_and_load(self):
        metric = Metric({"name": "logloss"})
        rf = ExtraTreesAlgorithm({"ml_task": "binary_classification"})
        rf.fit(self.X, self.y)
        y_predicted = rf.predict(self.X)
        loss = metric(self.y, y_predicted)

        filename = os.path.join(tempfile.gettempdir(), os.urandom(12).hex())

        rf.save(filename)
        rf2 = ExtraTreesAlgorithm({"ml_task": "binary_classification"})
        rf2.load(filename)
        # Finished with the file, delete it
        os.remove(filename)

        y_predicted = rf2.predict(self.X)
        loss2 = metric(self.y, y_predicted)
        assert_almost_equal(loss, loss2)

    def test_is_fitted(self):
        params = {"trees_in_step": 50, "ml_task": "binary_classification"}
        model = ExtraTreesAlgorithm(params)
        self.assertFalse(model.is_fitted())
        model.fit(self.X, self.y)
        self.assertTrue(model.is_fitted())
