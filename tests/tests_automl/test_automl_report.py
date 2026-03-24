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

        payload_dict = model.report_structured(format="dict", model_details=False)
        self.assertTrue(isinstance(payload_dict, dict))
        self.assertTrue("models" in payload_dict)
        self.assertGreater(len(payload_dict["models"]), 0)

        payload_json = model.report_structured(format="json")
        self.assertTrue(isinstance(payload_json, str))
        parsed = json.loads(payload_json)
        self.assertTrue("run_summary" in parsed)

        details_markdown = model.report_structured(model_details=True)
        self.assertTrue("## Model Details" in details_markdown)
        self.assertTrue("## 1_Baseline" in details_markdown)

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
