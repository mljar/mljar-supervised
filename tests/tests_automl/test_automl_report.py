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
