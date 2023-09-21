import os
import unittest
import tempfile
import json
import numpy as np
import pandas as pd
import shutil
from numpy.testing import assert_almost_equal

from sklearn import datasets
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA

from supervised import AutoML
from supervised.exceptions import AutoMLException

iris = datasets.load_iris()
housing = datasets.fetch_california_housing()
# limit data size for faster tests
housing.data = housing.data[:500]
housing.target = housing.target[:500]
breast_cancer = datasets.load_breast_cancer()


class AutoMLSampleWeightTest(unittest.TestCase):
    automl_dir = "automl_testing"

    def tearDown(self):
        shutil.rmtree(self.automl_dir, ignore_errors=True)

    def test_iris_dataset_sample_weight(self):
        """Tests AutoML in the iris dataset (Multiclass classification)
        without and with sample weight"""
        model = AutoML(
            explain_level=0, verbose=1, random_state=1, results_path=self.automl_dir
        )
        score_1 = model.fit(iris.data, iris.target).score(iris.data, iris.target)
        self.assertGreater(score_1, 0.5)

        shutil.rmtree(self.automl_dir, ignore_errors=True)
        model = AutoML(
            explain_level=0, verbose=1, random_state=1, results_path=self.automl_dir
        )
        sample_weight = np.ones(iris.data.shape[0])
        score_2 = model.fit(iris.data, iris.target, sample_weight=sample_weight).score(
            iris.data, iris.target, sample_weight=sample_weight
        )
        assert_almost_equal(score_1, score_2)

    def test_housing_dataset(self):
        """Tests AutoML in the housing dataset (Regression)
        without and with sample weight"""
        model = AutoML(
            explain_level=0, verbose=1, random_state=1, results_path=self.automl_dir
        )
        score_1 = model.fit(housing.data, housing.target).score(
            housing.data, housing.target
        )
        self.assertGreater(score_1, 0.5)

        shutil.rmtree(self.automl_dir, ignore_errors=True)
        model = AutoML(
            explain_level=0, verbose=1, random_state=1, results_path=self.automl_dir
        )
        sample_weight = np.ones(housing.data.shape[0])
        score_2 = model.fit(
            housing.data, housing.target, sample_weight=sample_weight
        ).score(housing.data, housing.target, sample_weight=sample_weight)
        assert_almost_equal(score_1, score_2)

    def test_breast_cancer_dataset(self):
        """Tests AutoML in the breast cancer (binary classification)
        without and with sample weight"""
        model = AutoML(
            explain_level=0, verbose=1, random_state=1, results_path=self.automl_dir
        )
        score_1 = model.fit(breast_cancer.data, breast_cancer.target).score(
            breast_cancer.data, breast_cancer.target
        )
        self.assertGreater(score_1, 0.5)

        shutil.rmtree(self.automl_dir, ignore_errors=True)
        model = AutoML(
            explain_level=0, verbose=1, random_state=1, results_path=self.automl_dir
        )
        sample_weight = np.ones(breast_cancer.data.shape[0])
        score_2 = model.fit(
            breast_cancer.data, breast_cancer.target, sample_weight=sample_weight
        ).score(breast_cancer.data, breast_cancer.target, sample_weight=sample_weight)
        assert_almost_equal(score_1, score_2)
