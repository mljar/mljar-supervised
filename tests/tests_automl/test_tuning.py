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

from supervised.algorithms.xgboost import additional
additional["max_rounds"] = 1

class AutoMLTuningTest(unittest.TestCase):

    automl_dir = "automl_tests"
    rows = 50

    def tearDown(self):
        shutil.rmtree(self.automl_dir, ignore_errors=True)

    def test_tune_only_default(self):
        X = np.random.rand(self.rows, 3)
        X = pd.DataFrame(X, columns=[f"f{i}" for i in range(3)])
        y = np.random.randint(0, 2, self.rows)

        automl = AutoML(
            results_path=self.automl_dir,
            total_time_limit=1,
            tuning_mode="Insane",
            algorithms=["Xgboost"],
        )

        automl.fit(X, y)
        self.assertEqual(len(automl._models), 1)
