import json
import importlib
import os
import shutil
import unittest
from io import StringIO
from unittest.mock import patch

import pandas as pd
from sklearn import datasets

from supervised import AutoML
from supervised.apps.templates import AI_DISCLOSURE_TEXT
from supervised.apps.metadata import _build_numeric_feature
from supervised.exceptions import AutoMLException


iris = datasets.load_iris()
digits = datasets.load_digits()
titanic = pd.read_csv("tests/data/Titanic/train.csv")


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
        self.assertEqual(manifest["title"], "MLJAR AutoML")
        self.assertEqual(manifest["default_notebook"], "predict_single.ipynb")
        self.assertEqual(len(manifest["notebooks"]), 2)
        self.assertEqual(manifest["model_task"], "multiclass_classification")
        self.assertEqual(len(manifest["feature_schema"]), iris.data.shape[1])
        self.assertEqual(manifest["python_requires"], ">=3.10")
        self.assertEqual(manifest["automl_bundle"]["archive_name"], "automl.zip")
        self.assertIn(f"App directory: {os.path.abspath(self.app_dir)}", stdout.getvalue())
        self.assertIn(
            f"Start Mercury: mercury --working-dir={os.path.abspath(self.app_dir)}",
            stdout.getvalue(),
        )

        with open(os.path.join(self.app_dir, "config.toml"), "r", encoding="utf-8") as fin:
            config_payload = fin.read()
        self.assertIn(AI_DISCLOSURE_TEXT, config_payload)

        with open(os.path.join(self.app_dir, "README.md"), "r", encoding="utf-8") as fin:
            readme_payload = fin.read()
        self.assertIn(AI_DISCLOSURE_TEXT, readme_payload)

        with open(os.path.join(self.app_dir, "predict_single.ipynb"), "r", encoding="utf-8") as fin:
            single_notebook = json.load(fin)
        self.assertEqual(single_notebook["cells"][-1]["cell_type"], "markdown")
        self.assertEqual(single_notebook["cells"][-1]["source"], AI_DISCLOSURE_TEXT)

        with open(os.path.join(self.app_dir, "predict_batch.ipynb"), "r", encoding="utf-8") as fin:
            batch_notebook = json.load(fin)
        self.assertEqual(batch_notebook["cells"][-1]["cell_type"], "markdown")
        self.assertEqual(batch_notebook["cells"][-1]["source"], AI_DISCLOSURE_TEXT)

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

        real_import_module = importlib.import_module

        def import_module(name):
            if name == "mercury":
                return object()
            return real_import_module(name)

        with patch("supervised.apps.generator.importlib.import_module", side_effect=import_module):
            with patch("sys.stdout", new_callable=StringIO) as stdout:
                model.app(verbose=True)

        output = stdout.getvalue()
        self.assertIn(f"App directory: {os.path.abspath(self.app_dir)}", output)
        self.assertIn(f"cd {os.path.abspath(self.app_dir)}", output)
        self.assertIn(
            f"Start Mercury: mercury --working-dir={os.path.abspath(self.app_dir)}",
            output,
        )
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

        real_import_module = importlib.import_module

        def import_module(name):
            if name == "mercury":
                raise ImportError("mercury not installed")
            return real_import_module(name)

        with patch(
            "supervised.apps.generator.importlib.import_module",
            side_effect=import_module,
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
        self.assertIn(
            f"Start Mercury: mercury --working-dir={os.path.abspath(self.app_dir)}",
            output,
        )

    def test_app_generates_batch_only_workspace_for_wide_datasets(self):
        model = AutoML(
            algorithms=["Baseline"],
            explain_level=0,
            verbose=0,
            random_state=1,
            results_path=self.automl_dir,
        )
        model.fit(digits.data, digits.target)

        with patch("sys.stdout", new_callable=StringIO) as stdout:
            output_path = model.app()

        self.assertEqual(output_path, os.path.abspath(self.app_dir))
        self.assertFalse(os.path.exists(os.path.join(self.app_dir, "predict_single.ipynb")))
        self.assertTrue(os.path.exists(os.path.join(self.app_dir, "predict_batch.ipynb")))

        with open(os.path.join(self.app_dir, "mljar_app.json"), "r") as fin:
            manifest = json.load(fin)

        self.assertEqual(manifest["default_notebook"], "predict_batch.ipynb")
        self.assertEqual(len(manifest["notebooks"]), 1)
        self.assertEqual(manifest["notebooks"][0]["filename"], "predict_batch.ipynb")
        self.assertFalse(manifest["supports"]["single_sample"])
        self.assertGreater(len(manifest["feature_schema"]), 15)
        output = stdout.getvalue()
        self.assertIn("Single prediction UI was skipped", output)
        self.assertIn("Generated app includes batch CSV prediction only.", output)

    def test_app_populates_select_choices_for_categorical_feature(self):
        model = AutoML(
            algorithms=["Baseline"],
            explain_level=0,
            verbose=0,
            random_state=1,
            results_path=self.automl_dir,
        )
        X = titanic[titanic.columns[2:]]
        y = titanic["Survived"]
        model.fit(X, y)

        with patch("sys.stdout", new_callable=StringIO):
            model.app()

        with open(os.path.join(self.app_dir, "mljar_app.json"), "r") as fin:
            manifest = json.load(fin)

        sex_feature = next(feature for feature in manifest["feature_schema"] if feature["name"] == "Sex")
        self.assertEqual(sex_feature["widget"], "select")
        self.assertIn("male", sex_feature["choices"])
        self.assertIn("female", sex_feature["choices"])

    def test_numeric_feature_uses_integer_step_for_integer_series(self):
        feature = _build_numeric_feature(
            {"name": "children", "kind": "numeric", "widget": "number"},
            pd.Series([0, 1, 2, 3, 4]),
        )

        self.assertEqual(feature["dtype"], "int")
        self.assertEqual(feature["step"], 1)
        self.assertEqual(feature["min"], 0)
        self.assertEqual(feature["max"], 4)

    def test_numeric_feature_treats_integer_like_float_series_as_int(self):
        feature = _build_numeric_feature(
            {"name": "age", "kind": "numeric", "widget": "number"},
            pd.Series([18.0, 25.0, 40.0, 67.0]),
        )

        self.assertEqual(feature["dtype"], "int")
        self.assertEqual(feature["step"], 1)
        self.assertEqual(feature["default"], 32)

    def test_numeric_feature_uses_decimal_step_for_float_series(self):
        feature = _build_numeric_feature(
            {"name": "fare", "kind": "numeric", "widget": "number"},
            pd.Series([7.25, 8.05, 12.5, 71.83]),
        )

        self.assertEqual(feature["dtype"], "float")
        self.assertEqual(feature["step"], 0.01)

    def test_numeric_feature_uses_integer_step_for_constant_numeric_series(self):
        feature = _build_numeric_feature(
            {"name": "dependents", "kind": "numeric", "widget": "number"},
            pd.Series([2.0, 2.0, 2.0]),
        )

        self.assertEqual(feature["dtype"], "int")
        self.assertEqual(feature["step"], 1)
