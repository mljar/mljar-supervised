import os
import unittest
import tempfile
import json
import numpy as np
import pandas as pd
import shutil
from sklearn import datasets

from supervised import AutoML


class EDATest(unittest.TestCase):

    automl_dir = "automl_1"

    def tearDown(self):
        shutil.rmtree(self.automl_dir, ignore_errors=True)

    def test_explain_default(self):
        a = AutoML(
            results_path=self.automl_dir,
            total_time_limit=10,
            algorithms=["Random Forest"],
            train_ensemble=False,
            explain_level=2,
        )

        X, y = datasets.make_classification(n_samples=100, n_features=5)

        X = pd.DataFrame(X, columns=[f"f_{i}" for i in range(X.shape[1])])
        y = pd.Series(y, name="class")

        a.fit(X, y)

        result_files = os.listdir(os.path.join(a._results_path, "EDA"))

        for col in X.columns:

            produced = True
            if "".join((col, ".png")) not in result_files:
                produced = False
                break
        self.assertTrue(produced)

        if "target.png" not in result_files:
            produced = False
        self.assertTrue(produced)

        if "README.md" not in result_files:
            produced = False
        self.assertTrue(produced)

        self.tearDown()
