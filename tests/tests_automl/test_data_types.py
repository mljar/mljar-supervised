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

class AutoMLDataTypesTest(unittest.TestCase):

    automl_dir = "automl_tests"
    rows = 250

    def tearDown(self):
        shutil.rmtree(self.automl_dir, ignore_errors=True)

    def test_bin_class_01(self):

        X = np.random.rand(self.rows, 3)
        X = pd.DataFrame(X, columns=[f"f{i}" for i in range(3)])
        y = np.random.randint(0, 2, self.rows)

        X["f1"] = X["f1"].astype("category")
        
        automl = AutoML(
            results_path=self.automl_dir,
            total_time_limit=1,
            algorithms=["CatBoost"],
            train_ensemble=False,
            explain_level=0,
            start_random_models=1,
        )
        automl.fit(X, y)
