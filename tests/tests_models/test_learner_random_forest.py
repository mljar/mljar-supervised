import unittest
import tempfile
import json
import numpy as np
import pandas as pd

from numpy.testing import assert_almost_equal
from sklearn import datasets

from supervised.models.learner_random_forest import RandomForestLearner
from supervised.metric import Metric


class RandomForestLearnerTest(unittest.TestCase):
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
        cls.data = {"train": {"X": cls.X, "y": cls.y}}

    def test_fit_predict(self):
        metric = Metric({"name": "logloss"})
        params = {"trees_in_step": 1}
        rf = RandomForestLearner(params)

        loss_prev = None
        for i in range(2):
            rf.fit(self.data["train"])
            y_predicted = rf.predict(self.X)
            loss = metric(self.y, y_predicted)
            if loss_prev is not None:
                self.assertTrue(loss + 0.00001 < loss_prev)
            loss_prev = loss

    def test_copy(self):
        metric = Metric({"name": "logloss"})
        rf = RandomForestLearner({})
        rf.fit(self.data["train"])
        y_predicted = rf.predict(self.X)
        loss = metric(self.y, y_predicted)

        rf2 = RandomForestLearner({})
        rf2 = rf.copy()
        self.assertEqual(type(rf), type(rf2))
        y_predicted = rf2.predict(self.X)
        loss2 = metric(self.y, y_predicted)
        assert_almost_equal(loss, loss2)

        rf.fit(self.data["train"])
        y_predicted = rf.predict(self.X)
        loss3 = metric(self.y, y_predicted)
        self.assertTrue(loss3 < loss)

        y_predicted = rf2.predict(self.X)
        loss4 = metric(self.y, y_predicted)
        assert_almost_equal(loss2, loss4)

    def test_save_and_load(self):
        metric = Metric({"name": "logloss"})
        rf = RandomForestLearner({})
        rf.fit(self.data["train"])
        y_predicted = rf.predict(self.X)
        loss = metric(self.y, y_predicted)

        json_desc = rf.save()
        rf2 = RandomForestLearner({})
        self.assertTrue(rf.uid != rf2.uid)
        rf2.load(json_desc)
        self.assertTrue(rf.uid == rf2.uid)

        y_predicted = rf2.predict(self.X)
        loss2 = metric(self.y, y_predicted)
        assert_almost_equal(loss, loss2)
