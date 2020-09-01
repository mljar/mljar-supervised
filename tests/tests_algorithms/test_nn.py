import unittest
import tempfile
import json
import numpy as np
import pandas as pd
import os

from numpy.testing import assert_almost_equal
from sklearn import datasets
from sklearn import preprocessing

from supervised.algorithms.nn import NeuralNetworkAlgorithm
from supervised.utils.metric import Metric


class NeuralNetworkAlgorithmTest(unittest.TestCase):
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
            "dense_layers": 2,
            "dense_1_size": 8,
            "dense_2_size": 4,
            "dropout": 0,
            "learning_rate": 0.01,
            "momentum": 0.9,
            "decay": 0.001,
            "ml_task": "binary_classification",
        }

    def test_fit_predict(self):
        metric = Metric({"name": "logloss"})
        nn = NeuralNetworkAlgorithm(self.params)
        loss_prev = None
        for _ in range(3):
            nn.fit(self.X, self.y)
            y_predicted = nn.predict(self.X)
            loss = metric(self.y, y_predicted)
            if loss_prev is not None:
                self.assertTrue(loss + 0.000001 < loss_prev)
            loss_prev = loss

    def test_copy(self):
        # train model #1
        metric = Metric({"name": "logloss"})
        nn = NeuralNetworkAlgorithm(self.params)
        nn.fit(self.X, self.y)
        y_predicted = nn.predict(self.X)
        loss = metric(self.y, y_predicted)
        # create model #2
        nn2 = NeuralNetworkAlgorithm(self.params)
        # model #2 is not initialized in constructor
        self.assertTrue(nn2.model is None)
        # do a copy and use it for predictions
        nn2 = nn.copy()
        self.assertEqual(type(nn), type(nn2))
        y_predicted = nn2.predict(self.X)
        loss2 = metric(self.y, y_predicted)
        self.assertEqual(loss, loss2)
        # fit model #1, there should be improvement in loss
        nn.fit(self.X, self.y)
        y_predicted = nn.predict(self.X)
        loss3 = metric(self.y, y_predicted)
        self.assertTrue(loss3 < loss)
        # the loss of model #2 should not change
        y_predicted = nn2.predict(self.X)
        loss4 = metric(self.y, y_predicted)
        assert_almost_equal(loss2, loss4)

    def test_save_and_load(self):
        metric = Metric({"name": "logloss"})
        nn = NeuralNetworkAlgorithm(self.params)
        nn.fit(self.X, self.y)
        y_predicted = nn.predict(self.X)
        loss = metric(self.y, y_predicted)

        filename = tempfile.gettempdir() + os.urandom(12).hex()

        nn.save(filename)
        json_desc = nn.get_params()
        nn2 = NeuralNetworkAlgorithm(json_desc["params"])
        nn2.load(filename)
        #Finished with the file, delete it 
        os.remove(filename)

        y_predicted = nn2.predict(self.X)
        loss2 = metric(self.y, y_predicted)
        assert_almost_equal(loss, loss2)


class RegressionNeuralNetworkAlgorithmTest(unittest.TestCase):
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
        nn = NeuralNetworkAlgorithm(self.params)
        loss_prev = None
        for _ in range(3):
            nn.fit(self.X, self.y)
            y_predicted = nn.predict(self.X)
            loss = metric(self.y, y_predicted)
            if loss_prev is not None:
                self.assertTrue(loss + 0.000001 < loss_prev)
            loss_prev = loss


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
        nn = NeuralNetworkAlgorithm(self.params)
        loss_prev = None
        for _ in range(3):
            nn.fit(self.X, self.y)
            y_predicted = nn.predict(self.X)
            loss = metric(self.y, y_predicted)
            if loss_prev is not None:
                self.assertTrue(loss + 0.000001 < loss_prev)
            loss_prev = loss
