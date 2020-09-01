import unittest
import tempfile
import json
import numpy as np
import pandas as pd
import os
from numpy.testing import assert_almost_equal
from sklearn import datasets

from supervised.algorithms.baseline import (
    BaselineClassifierAlgorithm,
    BaselineRegressorAlgorithm,
)
from supervised.utils.metric import Metric

import tempfile


class BaselineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, cls.y = datasets.make_regression(
            n_samples=100,
            n_features=5,
            n_informative=4,
            n_targets=1,
            shuffle=False,
            random_state=0,
        )

    def test_reproduce_fit_regression(self):
        metric = Metric({"name": "rmse"})
        prev_loss = None
        for _ in range(3):
            model = BaselineRegressorAlgorithm({"ml_task": "regression"})
            model.fit(self.X, self.y)
            y_predicted = model.predict(self.X)
            loss = metric(self.y, y_predicted)
            if prev_loss is not None:
                assert_almost_equal(prev_loss, loss)
            prev_loss = loss

    def test_reproduce_fit_bin_class(self):
        X, y = datasets.make_classification(
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
        metric = Metric({"name": "logloss"})
        prev_loss = None
        for _ in range(3):
            model = BaselineClassifierAlgorithm({"ml_task": "binary_classification"})
            model.fit(X, y)
            y_predicted = model.predict(X)
            loss = metric(y, y_predicted)
            if prev_loss is not None:
                assert_almost_equal(prev_loss, loss)
            prev_loss = loss

    def test_save_and_load(self):
        metric = Metric({"name": "rmse"})
        dt = BaselineRegressorAlgorithm({"ml_task": "regression"})
        dt.fit(self.X, self.y)
        y_predicted = dt.predict(self.X)
        loss = metric(self.y, y_predicted)

        backup_name = None
        try: 
            with tempfile.NamedTemporaryFile() as tmp:
                #Save and reload using temporary file
                dt.save(tmp.name)
                dt2 = BaselineRegressorAlgorithm({"ml_task": "regression"})
                dt2.load(tmp.name)
           
        #Catch Windows locking error from saving from within an existing NamedTemporaryFile.
        except PermissionError as e:
            #Try again in a new empty file
            backup_name = tmp.name + os.urandom(6).hex()
            dt.save(backup_name)
            dt2 = BaselineRegressorAlgorithm({"ml_task": "regression"})
            dt2.load(backup_name)

        #Ensure we clean up after ourselves.
        finally:
            if backup_name is not None:
                if os.path.exists(backup_name): os.remove(backup_name)
            if tmp.name is not None:
                if os.path.exists(tmp.name): os.remove(tmp.name)

        #Compare predictions between both models
        y_predicted = dt2.predict(self.X)
        loss2 = metric(self.y, y_predicted)
        assert_almost_equal(loss, loss2)