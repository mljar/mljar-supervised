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


class AutoMLRestoreTest(unittest.TestCase):

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
            total_time_limit=3,
            tuning_mode="Normal",
            algorithms=["Decision Tree"],
            explain_level=0,
            train_ensemble=False,
        )
        automl.fit(X, y)

        # Get number of starting models
        n1 = len([x for x in os.listdir(dir) if x[0].isdigit()])

        progress = json.load(open(os.path.join(self.automl_dir, "progress.json"), "r"))
        progress["fit_level"] = "default_algorithms"

        with open(os.path.join(self.automl_dir, "progress.json"), "w") as fout:
            fout.write(json.dumps(progress, indent=4))

        automl = AutoML(
            results_path=self.automl_dir,
            total_time_limit=3,
            tuning_mode="Normal",
            algorithms=["Decision Tree", "Xgboost"],
            explain_level=0,
            train_ensemble=False,
        )
        automl.fit(X, y)
        # Get number of models after second fit
        n2 = len([x for x in os.listdir(dir) if x[0].isdigit()])
        self.assertGreater(n2, n1)
