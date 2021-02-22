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

iris = datasets.load_iris()
boston = datasets.load_boston()
breast_cancer = datasets.load_breast_cancer()


class AutoMLTest(unittest.TestCase):

    automl_dir = "automl_testing"

    def tearDown(self):
        shutil.rmtree(self.automl_dir, ignore_errors=True)

    def setUp(self):
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
            a.predict(X)  # AutoML tries to load on predict call
        self.assertTrue("Cannot load" in str(context.exception))

    def test_get_params(self):
        """
        Passes params in AutoML constructor and uses `get_params()` after fitting.
        Initial params must be equal to the ones returned by `get_params()`.
        """
        # Create model
        model = AutoML(
            hill_climbing_steps=3, start_random_models=1, results_path=self.automl_dir
        )
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
        default_model = AutoML(
            algorithms=["Linear"], random_state=0, results_path=self.automl_dir
        )
        # Fit default model with transformed X and y, and predict transformed X
        y_pred_default = default_model.fit(new_X, y).predict(new_X)

        # Create pipeline with PCA and AutoML
        pipeline = make_pipeline(
            PCA(random_state=0), AutoML(algorithms=["Linear"], random_state=0)
        )
        # Fit with original X and y and predict X
        y_pred_pipe = pipeline.fit(X, y).predict(X)
        # y_pred_default must be equal to y_pred_pipe
        self.assertTrue((y_pred_pipe == y_pred_default).all())

    def test_predict_proba_in_regression(self):
        model = AutoML(
            explain_level=0, verbose=0, random_state=1, results_path=self.automl_dir
        )
        model.fit(boston.data, boston.target)
        with self.assertRaises(AutoMLException) as context:
            # Try to call predict_proba in regression task
            model.predict_proba(boston.data)

    def test_iris_dataset(self):
        """ Tests AutoML in the iris dataset (Multiclass classification)"""
        model = AutoML(
            explain_level=0, verbose=0, random_state=1, results_path=self.automl_dir
        )
        score = model.fit(iris.data, iris.target).score(iris.data, iris.target)
        self.assertGreater(score, 0.5)

    def test_boston_dataset(self):
        """ Tests AutoML in the boston dataset (Regression)"""
        model = AutoML(
            explain_level=0, verbose=0, random_state=1, results_path=self.automl_dir
        )
        score = model.fit(boston.data, boston.target).score(boston.data, boston.target)
        self.assertGreater(score, 0.5)

    def test_breast_cancer_dataset(self):
        """ Tests AutoML in the breast cancer (binary classification)"""
        model = AutoML(
            explain_level=0, verbose=0, random_state=1, results_path=self.automl_dir
        )
        score = model.fit(breast_cancer.data, breast_cancer.target).score(
            breast_cancer.data, breast_cancer.target
        )
        self.assertGreater(score, 0.5)

    def test_titatic_dataset(self):
        """ Tets AutoML in the titanic dataset (binary classification) with categorial features"""
        automl = AutoML(
            algorithms=["Xgboost"], mode="Explain", results_path=self.automl_dir
        )

        df = pd.read_csv("tests/data/Titanic/train.csv")

        X = df[df.columns[2:]]
        y = df["Survived"]

        automl.fit(X, y)

        test = pd.read_csv("tests/data/Titanic/test_with_Survived.csv")
        test_cols = [
            "Parch",
            "Ticket",
            "Fare",
            "Pclass",
            "Name",
            "Sex",
            "Age",
            "SibSp",
            "Cabin",
            "Embarked",
        ]
        score = automl.score(test[test_cols], test["Survived"])
        self.assertGreater(score, 0.5)

    def test_score_without_y(self):
        """Tests the use of `score()` without passing y. Should raise AutoMLException"""
        model = AutoML(
            explain_level=0, verbose=0, random_state=1, results_path=self.automl_dir
        )
        # Assert than an Exception is raised
        with self.assertRaises(AutoMLException) as context:
            # Try to score without passing 'y'
            score = model.fit(breast_cancer.data, breast_cancer.target).score(
                breast_cancer.data
            )

        self.assertTrue("y must be specified" in str(context.exception))

    def test_no_constructor_args(self):
        """Tests the use of AutoML without passing any args. Should work without any arguments"""
        # Create model with no arguments
        model = AutoML()
        model.results_path = self.automl_dir
        # Assert than an Exception is raised
        score = model.fit(iris.data, iris.target).score(iris.data, iris.target)
        self.assertGreater(score, 0.5)

    def test_fit_returns_self(self):
        """Tests if the `fit()` method returns `self`. This allows to quickly implement one-liners with AutoML"""
        model = AutoML()
        model.results_path = self.automl_dir
        self.assertTrue(
            isinstance(model.fit(iris.data, iris.target), AutoML),
            "`fit()` method must return 'self'",
        )

    def test_invalid_mode(self):
        model = AutoML(explain_level=0, verbose=0, results_path=self.automl_dir)
        param = {"mode": "invalid_mode"}
        model.set_params(**param)
        with self.assertRaises(ValueError) as context:
            model.fit(iris.data, iris.target)

    def test_invalid_ml_task(self):
        model = AutoML(explain_level=0, verbose=0, results_path=self.automl_dir)
        param = {"ml_task": "invalid_task"}
        model.set_params(**param)
        with self.assertRaises(ValueError) as context:
            model.fit(iris.data, iris.target)

    def test_invalid_results_path(self):
        model = AutoML(explain_level=0, verbose=0, results_path=self.automl_dir)
        param = {"results_path": 2}
        model.set_params(**param)
        with self.assertRaises(ValueError) as context:
            model.fit(iris.data, iris.target)

    def test_invalid_total_time_limit(self):
        model = AutoML(explain_level=0, verbose=0, results_path=self.automl_dir)
        param = {"total_time_limit": -1}
        model.set_params(**param)
        with self.assertRaises(ValueError) as context:
            model.fit(iris.data, iris.target)

    def test_invalid_model_time_limit(self):
        model = AutoML(explain_level=0, verbose=0, results_path=self.automl_dir)
        param = {"model_time_limit": -1}
        model.set_params(**param)
        with self.assertRaises(ValueError) as context:
            model.fit(iris.data, iris.target)

    def test_invalid_algorithm_name(self):
        model = AutoML(explain_level=0, verbose=0, results_path=self.automl_dir)
        param = {"algorithms": ["Baseline", "Neural Netrk"]}
        model.set_params(**param)
        with self.assertRaises(ValueError) as context:
            model.fit(iris.data, iris.target)

    def test_invalid_train_ensemble(self):
        model = AutoML(explain_level=0, verbose=0, results_path=self.automl_dir)
        param = {"train_ensemble": "not bool"}
        model.set_params(**param)
        with self.assertRaises(ValueError) as context:
            model.fit(iris.data, iris.target)

    def test_invalid_stack_models(self):
        model = AutoML(explain_level=0, verbose=0, results_path=self.automl_dir)
        param = {"stack_models": "not bool"}
        model.set_params(**param)
        with self.assertRaises(ValueError) as context:
            model.fit(iris.data, iris.target)

    def test_invalid_eval_metric(self):
        model = AutoML(explain_level=0, verbose=0, results_path=self.automl_dir)
        param = {"eval_metric": "not_real_metric"}
        model.set_params(**param)
        with self.assertRaises(ValueError) as context:
            model.fit(iris.data, iris.target)

    def test_invalid_validation_strategy(self):
        model = AutoML(explain_level=0, verbose=0, results_path=self.automl_dir)
        param = {"validation_strategy": "test"}
        model.set_params(**param)
        with self.assertRaises(ValueError) as context:
            model.fit(iris.data, iris.target)

    def test_invalid_verbose(self):
        model = AutoML(explain_level=0, verbose=0, results_path=self.automl_dir)
        param = {"verbose": -1}
        model.set_params(**param)
        with self.assertRaises(ValueError) as context:
            model.fit(iris.data, iris.target)

    def test_too_small_time_limit(self):
        rows = 100000
        X = np.random.uniform(size=(rows, 100))
        y = np.random.randint(0, 2, size=(rows,))

        automl = AutoML(results_path=self.automl_dir, total_time_limit=1, train_ensemble=False)
        with self.assertRaises(AutoMLException) as context:
            automl.fit(X, y)
