import unittest
import tempfile
import json
import numpy as np
import pandas as pd

from numpy.testing import assert_almost_equal
from sklearn import datasets
from supervised.automl import AutoML
from supervised.metric import Metric

import sklearn.model_selection
from sklearn.metrics import log_loss


class AutoMLWithRegressionTest(unittest.TestCase):
    def test_fit_and_predict(self):
        seed = 1709

        df = pd.read_csv("./tests/data/housing_regression_missing_values_missing_target.csv")
        print(df.columns)
        x_cols = [c for c in df.columns if c != "MEDV"]
        X = df[x_cols]
        y = df["MEDV"]

        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
            X, y, test_size=0.3, random_state=seed
        )
        automl = AutoML(
            total_time_limit=10,
            algorithms=["Xgboost"],  # ["LightGBM", "RF", "NN", "CatBoost", "Xgboost"],
            start_random_models=1,
            hill_climbing_steps=0,
            top_models_to_improve=0,
            train_ensemble=True,
            verbose=True,
        )
        automl.fit(X_train, y_train)

        response = automl.predict(X_test)  # ["p_1"]
        print("Response", response)

        # Compute the logloss on test dataset
        # ll = log_loss(y_test, response)
        # print("(*) Dataset id {} logloss {}".format(dataset_id, ll))


if __name__ == "__main__":
    unittest.main()
