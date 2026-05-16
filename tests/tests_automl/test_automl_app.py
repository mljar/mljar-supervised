import json
import os
import shutil
import unittest

from sklearn import datasets

from supervised import AutoML
from supervised.exceptions import AutoMLException


iris = datasets.load_iris()


class AutoMLAppTest(unittest.TestCase):
    automl_dir = "AutoMLAppTest"
    app_dir = "app"

    def setUp(self):
        shutil.rmtree(self.automl_dir, ignore_errors=True)
        shutil.rmtree(self.app_dir, ignore_errors=True)

    def tearDown(self):
        shutil.rmtree(self.automl_dir, ignore_errors=True)
        shutil.rmtree(self.app_dir, ignore_errors=True)

    def test_app_creates_default_workspace(self):
        model = AutoML(
            algorithms=["Baseline"],
            explain_level=0,
            verbose=0,
            random_state=1,
            results_path=self.automl_dir,
        )
        model.fit(iris.data, iris.target)

        output_path = model.app()

        self.assertEqual(output_path, os.path.abspath(self.app_dir))
        for path in [
            os.path.join(self.app_dir, "predict_single.ipynb"),
            os.path.join(self.app_dir, "predict_batch.ipynb"),
            os.path.join(self.app_dir, "app_support.py"),
            os.path.join(self.app_dir, "config.toml"),
            os.path.join(self.app_dir, "requirements.txt"),
            os.path.join(self.app_dir, "runtime.txt"),
            os.path.join(self.app_dir, "README.md"),
            os.path.join(self.app_dir, "mljar_app.json"),
            os.path.join(self.app_dir, "automl", "params.json"),
        ]:
            self.assertTrue(os.path.exists(path), path)

        with open(os.path.join(self.app_dir, "mljar_app.json"), "r") as fin:
            manifest = json.load(fin)

        self.assertEqual(manifest["bundle_type"], "automl_prediction_bundle")
        self.assertEqual(manifest["default_notebook"], "predict_single.ipynb")
        self.assertEqual(len(manifest["notebooks"]), 2)
        self.assertEqual(manifest["model_task"], "multiclass_classification")
        self.assertEqual(len(manifest["feature_schema"]), iris.data.shape[1])
        self.assertEqual(manifest["python_requires"], ">=3.10")

    def test_app_raises_when_output_exists(self):
        model = AutoML(
            algorithms=["Baseline"],
            explain_level=0,
            verbose=0,
            random_state=1,
            results_path=self.automl_dir,
        )
        model.fit(iris.data, iris.target)

        os.makedirs(self.app_dir)

        with self.assertRaises(AutoMLException):
            model.app()
