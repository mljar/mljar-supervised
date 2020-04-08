import os
import unittest
import tempfile
import json
import numpy as np
import pandas as pd
import shutil
from supervised import AutoML
from numpy.testing import assert_almost_equal
from sklearn import datasets
from supervised.exceptions import AutoMLException


class AutoMLValidationTest(unittest.TestCase):
    
    automl_dir = "automl_tests"

    def tearDown(self):
        shutil.rmtree(self.automl_dir)

    @classmethod
    def setUpClass(cls):
        cls.X, cls.y = datasets.make_classification(
            n_samples=200,
            n_features=5,
            n_informative=5,
            n_redundant=0,
            n_classes=2,
            n_clusters_per_class=1,
            n_repeated=0,
            shuffle=False,
            random_state=0,
        )
        cls.X = pd.DataFrame(cls.X, columns=["f0", "f1", "f2", "f3", "f4"])

    def test_set_validation(self):

        automl = AutoML(
            results_path=self.automl_dir,
            total_time_limit=1,
            algorithms=["Xgboost"],
            train_ensemble=False
        )
        automl._validation = {
            "validation_type": "kfold",
            "k_folds": 15,
            "shuffle": False,
            "stratify": True,
        }
        automl.fit(self.X, self.y)
