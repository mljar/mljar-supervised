import shutil
import unittest

import pandas as pd
from sklearn import datasets

from supervised import AutoML


class EnsembleSaveLoadTest(unittest.TestCase):
    automl_dir = "EnsembleSaveLoadTest"

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

        a2 = AutoML(results_path=self.automl_dir)
        p2 = a2.predict(X)

        self.assertTrue((p == p2).all())
