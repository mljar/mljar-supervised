import os
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


        #print(content)
        link = '<a href="1_Baseline/README.html">'
        self.assertFalse(link in content)



