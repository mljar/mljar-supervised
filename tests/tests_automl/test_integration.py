import os
import unittest
import tempfile
import json
import numpy as np
import pandas as pd
import shutil
from sklearn import datasets

from supervised import AutoML


class AutoMLIntegrationTest(unittest.TestCase):

    automl_dir = "automl_1"

    def tearDown(self):
        shutil.rmtree(self.automl_dir, ignore_errors=True)

    def test_integration(self):
        a = AutoML(results_path=self.automl_dir, total_time_limit=1, explain_level = 0)
        a.set_advanced(start_random_models=1)

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
        X = pd.DataFrame(X, columns=[f"f_{i}" for i in range(X.shape[1])])

        a.fit(X, y)
        p = a.predict(X)

        self.assertTrue("label" in p.columns)
