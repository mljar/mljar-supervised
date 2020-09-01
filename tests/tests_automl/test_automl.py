import os
import unittest
import tempfile
import json
import numpy as np
import pandas as pd
import shutil

from sklearn import datasets
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA

from supervised import AutoML
from supervised.exceptions import AutoMLException


class AutoMLTest(unittest.TestCase):

    automl_dir = "automl_testing"

    def tearDown(self):
        shutil.rmtree(self.automl_dir, ignore_errors=True)

    def test_new_directory(self):
        """ Directory does not exist, create it """
        # Assert directory does not exist
        self.assertTrue(not os.path.exists(self.automl_dir))
        # Create model with dir
        model = AutoML(results_path=self.automl_dir)
        # Generate data
        X, y = datasets.make_classification(n_samples=30)
        # Fit data
        model.fit(X, y)  # AutoML only validates constructor params on `fit()` call
        # Assert directory was created
        self.assertTrue(os.path.exists(self.automl_dir))

    def test_empty_directory(self):
        """ Directory exists and is empty, use it """
        # Assert directory does not exist
        self.assertTrue(not os.path.exists(self.automl_dir))
        # Make dir
        os.mkdir(self.automl_dir)
        # Assert dir exists
        self.assertTrue(os.path.exists(self.automl_dir))
        # Create automl with dir
        model = AutoML(results_path=self.automl_dir)
        # Generate data
        X, y = datasets.make_classification(n_samples=30)
        # Fit data
        model.fit(X, y)  # AutoML only validates constructor params on `fit()` call
        self.assertTrue(os.path.exists(self.automl_dir))

    def test_not_empty_directory(self):
        """
        Directory exists and is not empty,
        there is no params.json file in it, dont use it, raise exception
        """
        # Assert directory does not exist
        self.assertTrue(not os.path.exists(self.automl_dir))
        # Create directory
        os.mkdir(self.automl_dir)
        # Write some content to directory
        open(os.path.join(self.automl_dir, "test.file"), "w").close()
        # Assert directory exists
        self.assertTrue(os.path.exists(self.automl_dir))
        # Generate data
        X, y = datasets.make_classification(n_samples=30)
        # Assert than an Exception is raised
        with self.assertRaises(AutoMLException) as context:
            a = AutoML(results_path=self.automl_dir)
            a.fit(X, y)  # AutoML only validates constructor params on `fit()` call

        self.assertTrue("not empty" in str(context.exception))

    def test_use_directory_if_non_empty_exists_with_params_json(self):
        """
        Directory exists and is not empty,
        there is params.json in it, try to load it,
        raise exception because of fake params.json
        """
        # Assert directory does not exist
        self.assertTrue(not os.path.exists(self.automl_dir))
        # Create dir
        os.mkdir(self.automl_dir)
        # Write `params.json` to directory
        open(os.path.join(self.automl_dir, "params.json"), "w").close()
        # Assert directory exists
        self.assertTrue(os.path.exists(self.automl_dir))
        # Generate data
        X, y = datasets.make_classification(n_samples=30)
        with self.assertRaises(AutoMLException) as context:
            a = AutoML(results_path=self.automl_dir)
            a.fit(X, y)  # AutoML only validates constructor params on `fit()` call
        self.assertTrue("Cannot load" in str(context.exception))

    def test_get_params(self):
        """
        Passes params in AutoML constructor and uses `get_params()` after fitting.
        Initial params must be equal to the ones returned by `get_params()`.
        """
        # Create model
        model = AutoML(hill_climbing_steps=3, start_random_models=1)
        # Get params before fit
        params_before_fit = model.get_params()
        # Generate data
        X, y = datasets.make_classification(n_samples=30)
        # Fit data
        model.fit(X, y)
        # Get params after fit
        params_after_fit = model.get_params()
        # Assert before and after params are equal
        self.assertEquals(params_before_fit, params_after_fit)

    def test_scikit_learn_pipeline_integration(self):
        """
        Tests if AutoML is working on a scikit-learn's pipeline
        """
        # Create dataset
        X, y = datasets.make_classification(n_samples=30)
        # apply PCA to X
        new_X = PCA(random_state=0).fit_transform(X)
        # Create default model
        default_model = AutoML(algorithms=["Linear"], random_state=0)
        # Fit default model with transformed X and y, and predict transformed X
        y_pred_default = default_model.fit(new_X, y).predict(new_X)

        # Create pipeline with PCA and AutoML
        pipeline = make_pipeline(
            PCA(random_state=0), AutoML(algorithms=["Linear"], random_state=0)
        )
        # Fit with original X and y and predict X
        y_pred_pipe = pipeline.fit(X, y).predict(X)
        # y_pred_default must be equal to y_pred_pipe
        self.assertEquals(y_pred_pipe, y_pred_default)

    def test_iris_dataset(self):
        """ Tests AutoML in the iris dataset (Multiclass classification)"""
        iris = datasets.load_iris()
        model = AutoML(explain_level=0, verbose=0, random_state=1)
        score = model.fit(iris.data, iris.target).score(iris.data, iris.target)
        self.assertGreater(score, 0.5)

    def test_boston_dataset(self):
        """ Tests AutoML in the boston dataset (Regression)"""
        boston = datasets.load_boston()
        model = AutoML(explain_level=0, verbose=0, random_state=1)
        score = model.fit(boston.data, boston.target).score(boston.data, boston.target)
        self.assertGreater(score, 0.5)

    def test_breast_cancer_dataset(self):
        """ Tests AutoML in the boston dataset (Regression)"""
        breast_cancer = datasets.load_breast_cancer()
        model = AutoML(explain_level=0, verbose=0, random_state=1)
        score = model.fit(breast_cancer.data, breast_cancer.target).score(
            breast_cancer.data, breast_cancer.target
        )
        self.assertGreater(score, 0.5)
