import json
import os
import shutil
import unittest
from io import StringIO
from unittest.mock import patch

from sklearn import datasets

from supervised import AutoML
from supervised.exceptions import AutoMLException


iris = datasets.load_iris()


class AutoMLAppTest(unittest.TestCase):
    automl_dir = "AutoMLAppTest"
    app_dir = os.path.join(automl_dir, "app")

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

        with patch("sys.stdout", new_callable=StringIO) as stdout:
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
            os.path.join(self.app_dir, "automl.zip"),
        ]:
            self.assertTrue(os.path.exists(path), path)
        self.assertFalse(os.path.exists(os.path.join(self.app_dir, "automl")))

        with open(os.path.join(self.app_dir, "mljar_app.json"), "r") as fin:
            manifest = json.load(fin)

        self.assertEqual(manifest["bundle_type"], "automl_prediction_bundle")
        self.assertEqual(manifest["default_notebook"], "predict_single.ipynb")
        self.assertEqual(len(manifest["notebooks"]), 2)
        self.assertEqual(manifest["model_task"], "multiclass_classification")
        self.assertEqual(len(manifest["feature_schema"]), iris.data.shape[1])
        self.assertEqual(manifest["python_requires"], ">=3.10")
        self.assertEqual(manifest["automl_bundle"]["archive_name"], "automl.zip")
        self.assertIn(f"App directory: {os.path.abspath(self.app_dir)}", stdout.getvalue())
        self.assertIn("Start Mercury: mercury", stdout.getvalue())

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

    def test_app_verbose_false_prints_nothing(self):
        model = AutoML(
            algorithms=["Baseline"],
            explain_level=0,
            verbose=0,
            random_state=1,
            results_path=self.automl_dir,
        )
        model.fit(iris.data, iris.target)

        with patch("sys.stdout", new_callable=StringIO) as stdout:
            model.app(verbose=False)

        self.assertEqual(stdout.getvalue(), "")

    def test_app_verbose_prints_start_command_when_mercury_exists(self):
        model = AutoML(
            algorithms=["Baseline"],
            explain_level=0,
            verbose=0,
            random_state=1,
            results_path=self.automl_dir,
        )
        model.fit(iris.data, iris.target)

        with patch("supervised.apps.generator.importlib.import_module", return_value=object()):
            with patch("sys.stdout", new_callable=StringIO) as stdout:
                model.app(verbose=True)

        output = stdout.getvalue()
        self.assertIn(f"App directory: {os.path.abspath(self.app_dir)}", output)
        self.assertIn(f"cd {os.path.abspath(self.app_dir)}", output)
        self.assertIn("Start Mercury: mercury", output)
        self.assertNotIn("Mercury is not available.", output)

    def test_app_verbose_prints_install_message_when_mercury_import_fails(self):
        model = AutoML(
            algorithms=["Baseline"],
            explain_level=0,
            verbose=0,
            random_state=1,
            results_path=self.automl_dir,
        )
        model.fit(iris.data, iris.target)

        with patch(
            "supervised.apps.generator.importlib.import_module",
            side_effect=ImportError(),
        ):
            with patch("sys.stdout", new_callable=StringIO) as stdout:
                model.app(verbose=True)

        output = stdout.getvalue()
        self.assertIn(f"App directory: {os.path.abspath(self.app_dir)}", output)
        self.assertIn(f"cd {os.path.abspath(self.app_dir)}", output)
        self.assertIn(
            "Mercury is not available in the current Python environment. "
            "Install it with: pip install -r requirements.txt",
            output,
        )
        self.assertIn("Start Mercury: mercury", output)
