import unittest
import tempfile
import json
import numpy as np
import pandas as pd

from numpy.testing import assert_almost_equal
from sklearn import datasets

from supervised.algorithms.knn import KNeighborsAlgorithm, KNeighborsRegressorAlgorithm
from supervised.utils.metric import Metric

import tempfile


class KNeighborsRegressorAlgorithmTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, cls.y = datasets.make_regression(
            n_samples=100,
            n_features=5, 
            n_informative=4, 
            shuffle=False, 
            random_state=0
        )

    def test_reproduce_fit(self):
        metric = Metric({"name": "mse"})
        params = {"seed": 1, "ml_task": "regression"}
        prev_loss = None
        for _ in range(2):
            model = KNeighborsRegressorAlgorithm(params)
            model.fit(self.X, self.y)
            y_predicted = model.predict(self.X)
            loss = metric(self.y, y_predicted)
            if prev_loss is not None:
                assert_almost_equal(prev_loss, loss)
            prev_loss = loss


class KNeighborsAlgorithmTest(unittest.TestCase):
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
        for _ in range(2):
            model = KNeighborsAlgorithm(params)
            model.fit(self.X, self.y)
            y_predicted = model.predict(self.X)
            loss = metric(self.y, y_predicted)
            if prev_loss is not None:
                assert_almost_equal(prev_loss, loss)
            prev_loss = loss

    def test_fit_predict(self):
        metric = Metric({"name": "logloss"})
        params = {"ml_task": "binary_classification"}
        la = KNeighborsAlgorithm(params)

        la.fit(self.X, self.y)
        y_predicted = la.predict(self.X)
        self.assertTrue(metric(self.y, y_predicted) < 0.6)

    def test_is_fitted(self):
        params = {"ml_task": "binary_classification"}
        model = KNeighborsAlgorithm(params)
        self.assertFalse(model.is_fitted())
        model.fit(self.X, self.y)
        self.assertTrue(model.is_fitted())

    def test_classes_attribute(self):
        params = {"ml_task": "binary_classification"}
        model = KNeighborsAlgorithm(params)
        model.fit(self.X,self.y)


        try:
            classes = model._classes  
        except AttributeError:
            classes = None

        #debug statements
        print("np.unique(self.y):", np.unique(self.y))
        print("classes:", classes)

        self.assertTrue(np.array_equal(np.unique(self.y), classes))
