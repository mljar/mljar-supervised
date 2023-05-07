import unittest
import shutil
import numpy as np
import pandas as pd
from sklearn import datasets
from supervised import AutoML
import os
import json


class EnsembleSaveLoadTest(unittest.TestCase):

    def setUp(self):
        self.automl_dir = "automl_01"

    def tearDown(self):
        shutil.rmtree(self.automl_dir, ignore_errors=True)

    def test_save_load(self):
        a = AutoML(
            results_path=self.automl_dir,
            total_time_limit=10,
            explain_level=0,
            mode="Explain",
            train_ensemble=True,
            start_random_models=1,
        )

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

        # Find framework.json
        framework_paths = []
        for model_subpath in a._model_subpaths:
            framework_path = f"{self.automl_dir}/{model_subpath}/framework.json"
            framework_paths.append(framework_path)

        # Copy first joblib version in framework.json
        with open(framework_paths[0], "r") as f:
            framework_data = json.load(f)
            if isinstance(framework_data, list):
                framework_data = framework_data[0]
            expected_joblib_version = framework_data["joblib_version"]

        a2 = AutoML()
        a2.load(self.automl_dir, expected_joblib_version=expected_joblib_version)
        p2 = a2.predict(X)

        self.assertTrue((p == p2).all())

if __name__ == '__main__':
    unittest.main()
