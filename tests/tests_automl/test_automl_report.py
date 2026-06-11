import os
import json
import shutil
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline

from supervised import AutoML
from supervised.exceptions import AutoMLException
from supervised.fairness.certificate import build_certificate_info
from supervised.utils.report_structured import AUTOMATION_DISCLOSURE_TEXT

iris = datasets.load_iris()

class AutoMLReportTest(unittest.TestCase):
    automl_dir = "AutoMLTest"

    def tearDown(self):
        shutil.rmtree(self.automl_dir, ignore_errors=True)

    def setUp(self):
        shutil.rmtree(self.automl_dir, ignore_errors=True)

    def test_report(self):
        """Tests AutoML in the iris dataset (Multiclass classification)"""
        model = AutoML(
            algorithms=["Baseline"],
            explain_level=0, verbose=0, random_state=1, results_path=self.automl_dir
        )
        model.fit(iris.data, iris.target)
        model.report()

        report_path = os.path.join(self.automl_dir, "README.html")
        self.assertTrue(os.path.exists(report_path))

        content = None
        with open(report_path, "r") as fin:
            content = fin.read()
        self.assertTrue("AutoML Leaderboard" in content)
        self.assertIn(AUTOMATION_DISCLOSURE_TEXT, content)

    def test_report_structured(self):
        model = AutoML(
            algorithms=["Baseline"],
            explain_level=0,
            verbose=0,
            random_state=1,
            results_path=self.automl_dir,
        )
        model.fit(iris.data, iris.target)

        markdown_report = model.report_structured()
        self.assertTrue(isinstance(markdown_report, str))
        self.assertTrue("# MLJAR AutoML Report" in markdown_report)
        self.assertIn(AUTOMATION_DISCLOSURE_TEXT, markdown_report)

        report_json_path = os.path.join(self.automl_dir, "report_structured.json")
        self.assertTrue(os.path.exists(report_json_path))

        with open(report_json_path, "r") as fin:
            payload = json.load(fin)

        for key in [
            "created_at_utc",
            "mljar_supervised_version",
            "run_summary",
            "leaderboard",
            "best_model",
            "models",
            "artifacts",
        ]:
            self.assertTrue(key in payload)

        self.assertTrue(isinstance(payload["models"], list))
        self.assertGreater(len(payload["models"]), 0)

        payload_dict = model.report_structured(format="dict")
        self.assertTrue(isinstance(payload_dict, dict))
        self.assertTrue("leaderboard" in payload_dict)
        self.assertTrue("global_feature_importance" in payload_dict)
        self.assertFalse("best_model" in payload_dict)
        self.assertFalse("models" in payload_dict)
        self.assertFalse("run_summary" in payload_dict)

        payload_json = model.report_structured(format="json")
        self.assertTrue(isinstance(payload_json, str))
        parsed = json.loads(payload_json)
        self.assertTrue("leaderboard" in parsed)
        self.assertTrue("global_feature_importance" in parsed)
        self.assertFalse("best_model" in parsed)
        self.assertFalse("models" in parsed)

        gfi = payload_dict.get("global_feature_importance", {})
        if gfi.get("available"):
            n = gfi.get("n_features", 0)
            k = gfi.get("selection_k", 0)
            if n < 10:
                self.assertEqual(k, min(3, n))
            elif n < 20:
                self.assertEqual(k, min(5, n))
            else:
                self.assertEqual(k, min(10, n))
            self.assertEqual(len(gfi.get("top", [])), k)
            self.assertEqual(len(gfi.get("bottom", [])), k)
            self.assertTrue("## Global Feature Importance (Averaged Across Models)" in markdown_report)

        details_markdown = model.report_structured(model_name="1_Baseline")
        self.assertTrue("# MLJAR AutoML report for 1_Baseline" in details_markdown)
        self.assertIn(AUTOMATION_DISCLOSURE_TEXT, details_markdown)
        self.assertTrue("## Hyperparameters" in details_markdown)

        details_dict = model.report_structured(format="dict", model_name="1_Baseline")
        self.assertTrue("selected_model" in details_dict)
        self.assertEqual(details_dict["selected_model"]["name"], "1_Baseline")
        self.assertTrue("hyperparameters" in details_dict["selected_model"])
        self.assertTrue(isinstance(details_dict["selected_model"]["hyperparameters"], dict))

        with self.assertRaises(AutoMLException):
            model.report_structured(model_name="not_existing_model")

        for m in payload.get("models", []):
            fi = m.get("feature_importance", {})
            if not fi.get("available"):
                continue
            n = fi.get("n_features", 0)
            k = fi.get("selection_k", 0)
            if n < 10:
                self.assertEqual(k, min(3, n))
            elif n < 20:
                self.assertEqual(k, min(5, n))
            else:
                self.assertEqual(k, min(10, n))
            self.assertEqual(len(fi.get("top", [])), k)
            self.assertEqual(len(fi.get("worst", [])), k)

    def test_fairness_certificate_in_reports(self):
        X = np.random.uniform(size=(30, 2))
        y = np.random.randint(0, 2, size=(30,))
        sensitive = pd.DataFrame({"sensitive": ["A", "B"] * 15})

        model = AutoML(
            results_path=self.automl_dir,
            model_time_limit=10,
            algorithms=["Xgboost"],
            fairness_metric="equalized_odds_difference",
            fairness_threshold=1.1,
            explain_level=0,
            verbose=0,
            train_ensemble=False,
            stack_models=False,
            validation_strategy={"validation_type": "split"},
            start_random_models=1,
        )
        model.fit(X, y, sensitive_features=sensitive)
        model.report()

        main_readme = Path(self.automl_dir) / "README.md"
        self.assertTrue(main_readme.exists())
        main_content = main_readme.read_text()
        self.assertIn("## Fairness Certificate", main_content)
        self.assertIn("https://mljar.com/fairness-certificate/", main_content)

        best_model_name = model._best_model.get_name()
        model_readme = Path(self.automl_dir) / best_model_name / "README.md"
        self.assertTrue(model_readme.exists())
        self.assertIn(
            "https://mljar.com/fairness-certificate/",
            model_readme.read_text(),
        )

        payload_dict = model.report_structured(format="dict")
        fairness_summary = payload_dict.get("fairness_summary")
        self.assertTrue(isinstance(fairness_summary, dict))
        self.assertIn("certificate_url", fairness_summary)
        self.assertIn("certificate_params", fairness_summary)
        self.assertEqual(
            fairness_summary["certificate_params"]["status"],
            "PASSED",
        )

        selected_dict = model.report_structured(format="dict", model_name=best_model_name)
        selected_fairness = selected_dict["selected_model"]["fairness"]
        self.assertIn("certificate_url", selected_fairness)
        self.assertIn("certificate_params", selected_fairness)

        selected_markdown = model.report_structured(model_name=best_model_name)
        self.assertIn("## Fairness Summary", selected_markdown)
        self.assertIn("### Fairness Certificate", selected_markdown)

    def test_certificate_info_formats_multiclass_feature(self):
        fairness_details = {
            "sensitive__approved": {
                "fairness_metric_name": "Demographic Parity Ratio",
                "fairness_metric_value": 0.91,
                "fairness_threshold": 0.8,
                "is_fair": True,
            }
        }

        certificate = build_certificate_info(
            "Model",
            "multiclass_classification",
            fairness_details,
            worst_fairness=0.91,
            is_fair=True,
            issue_date="2026-05-18",
        )

        self.assertEqual(
            certificate["certificate_params"]["sensitiveFeature"],
            "sensitive (class: approved)",
        )

    def test_certificate_info_uses_first_matching_tie(self):
        fairness_details = {
            "first": {
                "fairness_metric_name": "Demographic Parity Ratio",
                "fairness_metric_value": 0.7,
                "fairness_threshold": 0.6,
                "is_fair": True,
            },
            "second": {
                "fairness_metric_name": "Demographic Parity Ratio",
                "fairness_metric_value": 0.7,
                "fairness_threshold": 0.6,
                "is_fair": True,
            },
        }

        certificate = build_certificate_info(
            "Model",
            "binary_classification",
            fairness_details,
            worst_fairness=0.7,
            is_fair=True,
            issue_date="2026-05-18",
        )

        self.assertEqual(
            certificate["certificate_params"]["sensitiveFeature"],
            "first",
        )
