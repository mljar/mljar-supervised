import os
import shutil
import unittest
import numpy as np
import pandas as pd
import pytest
from sklearn import datasets
from supervised import AutoML
from supervised.exceptions import AutoMLException

class AutoMLPredictUncertaintyTest(unittest.TestCase):
    automl_dir = "AutoML_Predict_Uncertainty_Test"

    def tearDown(self):
        shutil.rmtree(self.automl_dir, ignore_errors=True)

    def setUp(self):
        shutil.rmtree(self.automl_dir, ignore_errors=True)

    def test_predict_uncertainty_before_fit(self):
        """Should raise AutoMLException when called before fit"""
        automl = AutoML(results_path=self.automl_dir)
        X, y = datasets.make_regression(n_samples=50, n_features=5, random_state=42)
        with self.assertRaises(AutoMLException) as context:
            automl.predict_uncertainty(X)
        self.assertIn("has not been fitted yet", str(context.exception))

    def test_predict_uncertainty_classification(self):
        """Should raise AutoMLException for classification tasks"""
        X, y = datasets.make_classification(n_samples=50, n_features=5, n_classes=2, random_state=42)
        automl = AutoML(
            results_path=self.automl_dir,
            algorithms=["Decision Tree"],
            explain_level=0,
            verbose=0
        )
        automl.fit(X, y)
        with self.assertRaises(AutoMLException) as context:
            automl.predict_uncertainty(X)
        self.assertIn("only supported for regression tasks", str(context.exception))

    def test_predict_uncertainty_no_ensemble(self):
        """Should raise AutoMLException when the best model is not an Ensemble and no Ensemble was trained"""
        X, y = datasets.make_regression(n_samples=50, n_features=5, random_state=42)
        automl = AutoML(
            results_path=self.automl_dir,
            algorithms=["Decision Tree"],
            train_ensemble=False,
            explain_level=0,
            verbose=0,
            random_state=42
        )
        automl.fit(X, y)
        
        self.assertNotEqual(automl._best_model.get_type(), "Ensemble")
        
        with self.assertRaises(AutoMLException) as context:
            automl.predict_uncertainty(X)
        self.assertIn("Ensemble model is available", str(context.exception))

    def test_predict_uncertainty_ensemble_fallback(self):
        """Should fallback to Ensemble model if the best model is not an Ensemble but Ensemble was trained"""
        X, y = datasets.make_regression(n_samples=50, n_features=5, random_state=42)
        automl = AutoML(
            results_path=self.automl_dir,
            algorithms=["Linear", "Decision Tree"],
            train_ensemble=True,
            explain_level=0,
            verbose=0,
            random_state=42
        )
        automl.fit(X, y)
        
        for m in automl._models:
            if m.get_type() == "Decision Tree":
                automl._best_model = m
                break
        
        self.assertEqual(automl._best_model.get_type(), "Decision Tree")
        
        res = automl.predict_uncertainty(X)
        self.assertIsInstance(res, pd.DataFrame)
        self.assertIn("prediction", res.columns)

    def test_predict_uncertainty_regression_success(self):
        """Should successfully compute uncertainty for regression when ensemble is available"""
        np.random.seed(42)
        X = np.random.rand(120, 5)
        y = X[:, 0]**2 + np.sin(X[:, 1] * np.pi) + np.exp(X[:, 2]) + np.random.normal(0, 0.05, 120)
        X = pd.DataFrame(X, columns=[f"f_{i}" for i in range(5)])

        automl = AutoML(
            results_path=self.automl_dir,
            algorithms=["Linear", "Decision Tree", "Random Forest"],
            explain_level=0,
            verbose=0,
            random_state=42
        )
        automl.fit(X, y)
        
        self.assertEqual(automl._best_model.get_type(), "Ensemble")

        alpha = 0.05
        res = automl.predict_uncertainty(X, alpha=alpha)
        
        # Assertions
        self.assertIsInstance(res, pd.DataFrame)
        
        expected_cols = ["prediction", "prediction_std", "prediction_variance", "lower", "upper"]
        for col in expected_cols:
            self.assertIn(col, res.columns)
            
        self.assertEqual(len(res), len(X))
        
        # Check standard deviation and variance are non-negative
        self.assertTrue((res["prediction_std"] >= 0).all())
        self.assertTrue((res["prediction_variance"] >= 0).all())
        
        # Check lower <= prediction <= upper
        self.assertTrue((res["lower"] <= res["prediction"]).all())
        self.assertTrue((res["prediction"] <= res["upper"]).all())
        
        # Check standard deviation calculation is close to square root of variance
        np.testing.assert_array_almost_equal(res["prediction_std"], np.sqrt(res["prediction_variance"]))
