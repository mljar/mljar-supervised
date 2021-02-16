import unittest
import tempfile
import json
import numpy as np
import pandas as pd
import os

from numpy.testing import assert_almost_equal
from sklearn import datasets
from sklearn import preprocessing

from supervised.algorithms.nn import MLPAlgorithm, MLPRegressorAlgorithm
from supervised.utils.metric import Metric


class MLPAlgorithmTest(unittest.TestCase):
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
            random_state=1,
        )

        cls.params = {
            "dense_1_size": 8,
            "dense_2_size": 4,
            "learning_rate": 0.01,
            "ml_task": "binary_classification",
        }

    def test_fit_predict(self):
        metric = Metric({"name": "logloss"})
        nn = MLPAlgorithm(self.params)
        nn.fit(self.X, self.y)
        y_predicted = nn.predict_proba(self.X)
        loss = metric(self.y, y_predicted)
        self.assertLess(loss, 2)

    def test_copy(self):
        # train model #1
        metric = Metric({"name": "logloss"})
        nn = MLPAlgorithm(self.params)
        nn.fit(self.X, self.y)
        y_predicted = nn.predict(self.X)
        loss = metric(self.y, y_predicted)
        # create model #2
        nn2 = MLPAlgorithm(self.params)
        # do a copy and use it for predictions
        nn2 = nn.copy()
        self.assertEqual(type(nn), type(nn2))
        y_predicted = nn2.predict(self.X)
        loss2 = metric(self.y, y_predicted)
        self.assertEqual(loss, loss2)

        # the loss of model #2 should not change
        y_predicted = nn2.predict(self.X)
        loss4 = metric(self.y, y_predicted)
        assert_almost_equal(loss2, loss4)

    def test_save_and_load(self):
        metric = Metric({"name": "logloss"})
        nn = MLPAlgorithm(self.params)
        nn.fit(self.X, self.y)
        y_predicted = nn.predict(self.X)
        loss = metric(self.y, y_predicted)

        filename = os.path.join(tempfile.gettempdir(), os.urandom(12).hex())

        nn.save(filename)
        json_desc = nn.get_params()
        nn2 = MLPAlgorithm(json_desc["params"])
        nn2.load(filename)
        # Finished with the file, delete it
        os.remove(filename)

        y_predicted = nn2.predict(self.X)
        loss2 = metric(self.y, y_predicted)
        assert_almost_equal(loss, loss2)


class MLPRegressorAlgorithmTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, cls.y = datasets.make_regression(
            n_samples=100, n_features=5, n_informative=4, shuffle=False, random_state=0
        )

        cls.params = {
            "dense_layers": 2,
            "dense_1_size": 8,
            "dense_2_size": 4,
            "dropout": 0,
            "learning_rate": 0.01,
            "momentum": 0.9,
            "decay": 0.001,
            "ml_task": "regression",
        }

        cls.y = preprocessing.scale(cls.y)

    def test_fit_predict(self):
        metric = Metric({"name": "mse"})
        nn = MLPRegressorAlgorithm(self.params)
        nn.fit(self.X, self.y)
        y_predicted = nn.predict(self.X)
        loss = metric(self.y, y_predicted)
        self.assertLess(loss, 2)


class MultiClassNeuralNetworkAlgorithmTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, cls.y = datasets.make_classification(
            n_samples=100,
            n_features=5,
            n_informative=4,
            n_redundant=1,
            n_classes=3,
            n_clusters_per_class=3,
            n_repeated=0,
            shuffle=False,
            random_state=0,
        )

        cls.params = {
            "dense_layers": 2,
            "dense_1_size": 8,
            "dense_2_size": 4,
            "dropout": 0,
            "learning_rate": 0.01,
            "momentum": 0.9,
            "decay": 0.001,
            "ml_task": "multiclass_classification",
            "num_class": 3,
        }

        lb = preprocessing.LabelBinarizer()
        lb.fit(cls.y)
        cls.y = lb.transform(cls.y)

    def test_fit_predict(self):
        metric = Metric({"name": "logloss"})
        nn = MLPAlgorithm(self.params)
        nn.fit(self.X, self.y)
        y_predicted = nn.predict(self.X)
        loss = metric(self.y, y_predicted)
        self.assertLess(loss, 2)

    def test_is_fitted(self):
        model = MLPAlgorithm(self.params)
        self.assertFalse(model.is_fitted())
        model.fit(self.X, self.y)
        self.assertTrue(model.is_fitted())