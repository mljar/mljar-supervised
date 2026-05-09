import os
import shutil
import unittest

import pandas as pd
import pytest
from sklearn import datasets

from supervised import AutoML


iris = datasets.load_iris()


@pytest.mark.usefixtures("data_folder")
class AutoMLFeatureImportanceApiTest(unittest.TestCase):
    automl_dir = "AutoMLFeatureImportanceApiTest"

    def tearDown(self):
        shutil.rmtree(self.automl_dir, ignore_errors=True)

    def setUp(self):
        shutil.rmtree(self.automl_dir, ignore_errors=True)

    def _train(self):
        automl = AutoML(
            results_path=self.automl_dir,
            mode="Explain",
            explain_level=1,
            algorithms=["Decision Tree"],
            train_ensemble=False,
            random_state=1,
            verbose=0,
        )
        automl.fit(iris.data, iris.target)
        return automl

    def test_get_feature_importance_best(self):
        automl = self._train()
        fi = automl.get_feature_importance()
        self.assertTrue(isinstance(fi, pd.DataFrame))
        self.assertTrue("feature" in fi.columns)
        self.assertTrue("importance" in fi.columns)
        self.assertGreater(fi.shape[0], 0)

    def test_get_feature_importance_all_and_model_name(self):
        automl = self._train()
        fi_all = automl.get_feature_importance(model="all")
        self.assertTrue(isinstance(fi_all, dict))
        self.assertTrue(automl._best_model.get_name() in fi_all)

        fi_model = automl.get_feature_importance(model=automl._best_model.get_name())
        self.assertTrue(isinstance(fi_model, pd.DataFrame))
        self.assertGreater(fi_model.shape[0], 0)

    def test_get_feature_importance_normalized(self):
        automl = self._train()
        fi = automl.get_feature_importance(kind="normalized")
        self.assertGreater(fi.shape[0], 0)
        self.assertGreaterEqual(fi["importance"].min(), 0.0)
        self.assertLessEqual(fi["importance"].max(), 1.0)

    def test_recompute_feature_importance_if_missing(self):
        automl = self._train()
        model_name = automl._best_model.get_name()
        model_path = os.path.join(self.automl_dir, model_name)

        for f in os.listdir(model_path):
            if "_importance.csv" in f and "shap" not in f:
                os.remove(os.path.join(model_path, f))

        fi = automl.get_feature_importance(model=model_name)
        self.assertTrue(isinstance(fi, pd.DataFrame))
        self.assertTrue("feature" in fi.columns)
        self.assertTrue("importance" in fi.columns)
        self.assertGreater(fi.shape[0], 0)
