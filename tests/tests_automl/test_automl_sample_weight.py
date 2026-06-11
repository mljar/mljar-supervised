import shutil
import unittest

import numpy as np
from sklearn import datasets

from supervised import AutoML

iris = datasets.load_iris()
housing = datasets.make_regression(
    n_samples=500,
    n_features=8,
    n_informative=8,
    noise=5.0,
    random_state=123,
)
breast_cancer = datasets.load_breast_cancer()


class AutoMLSampleWeightTest(unittest.TestCase):
    automl_dir = "AutoMLSampleWeightTest"
    score_tolerance = 0.1

    def tearDown(self):
        shutil.rmtree(self.automl_dir, ignore_errors=True)

    def assert_score_close(self, score_1, score_2):
        self.assertLessEqual(abs(score_1 - score_2), self.score_tolerance)

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
        self.assert_score_close(score_1, score_2)

    def test_housing_dataset(self):
        """Tests AutoML in the housing dataset (Regression)
        without and with sample weight"""
        model = AutoML(
            explain_level=0, verbose=1, random_state=1, results_path=self.automl_dir
        )
        score_1 = model.fit(housing[0], housing[1]).score(
            housing[0], housing[1]
        )
        self.assertGreater(score_1, 0.5)

        shutil.rmtree(self.automl_dir, ignore_errors=True)
        model = AutoML(
            explain_level=0, verbose=1, random_state=1, results_path=self.automl_dir
        )
        sample_weight = np.ones(housing[0].shape[0])
        score_2 = model.fit(
            housing[0], housing[1], sample_weight=sample_weight
        ).score(housing[0], housing[1], sample_weight=sample_weight)
        self.assert_score_close(score_1, score_2)

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
        self.assert_score_close(score_1, score_2)
