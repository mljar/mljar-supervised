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

    def test_reproduce_fit(self):
        metric = Metric({"name": "logloss"})
        params = {"trees_in_step": 1, "seed": 1}
        prev_loss = None
        for i in range(3):
            model = RandomForestLearner(params)
            model.fit(self.X, self.y)
            y_predicted = model.predict(self.X)
            loss = metric(self.y, y_predicted)
            if prev_loss is not None:
                assert_almost_equal(prev_loss, loss)
            prev_loss = loss

    def test_fit_predict(self):
        metric = Metric({"name": "logloss"})
        params = {"trees_in_step": 50}
        rf = RandomForestLearner(params)

        rf.fit(self.X, self.y)
        y_predicted = rf.predict(self.X)
        self.assertTrue(metric(self.y, y_predicted) < 0.6)

    def test_copy(self):
        metric = Metric({"name": "logloss"})
        rf = RandomForestLearner({})
        rf.fit(self.X, self.y)
        y_predicted = rf.predict(self.X)
        loss = metric(self.y, y_predicted)

        rf2 = RandomForestLearner({})
        rf2 = rf.copy()
        self.assertEqual(type(rf), type(rf2))
        y_predicted = rf2.predict(self.X)
        loss2 = metric(self.y, y_predicted)
        assert_almost_equal(loss, loss2)

        rf.fit(self.X, self.y)
        y_predicted = rf.predict(self.X)
        loss3 = metric(self.y, y_predicted)
        self.assertNotEqual(loss3, loss)

        y_predicted = rf2.predict(self.X)
        loss4 = metric(self.y, y_predicted)
        assert_almost_equal(loss2, loss4)

    def test_save_and_load(self):
        metric = Metric({"name": "logloss"})
        rf = RandomForestLearner({})
        rf.fit(self.X, self.y)
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


if __name__ == "__main__":
    unittest.main()
