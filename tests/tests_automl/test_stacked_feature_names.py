import unittest

import pandas as pd

from supervised import AutoML
from supervised.algorithms.registry import BINARY_CLASSIFICATION


class _ModelMock:
    def __init__(self, name, oof_df, pred_df):
        self._name = name
        self._oof = oof_df
        self._pred = pred_df

    def get_name(self):
        return self._name

    def get_out_of_folds(self):
        return self._oof.copy()

    def predict(self, X):
        out = self._pred.copy()
        out.index = X.index
        return out


class AutoMLStackedFeatureNamesTest(unittest.TestCase):
    def test_binary_stacked_feature_names_are_consistent(self):
        X = pd.DataFrame({"f1": [0.1, 0.2], "f2": [1.0, 2.0]})

        # Training OOF has a single positive-class probability column
        # with non-0/1 class naming convention.
        oof = pd.DataFrame(
            {"prediction_0_for_no_1_for_yes": [0.7, 0.8], "target": [1, 0]}
        )
        # Predict-time output can have two probability columns and label.
        pred = pd.DataFrame(
            {
                "prediction_0_for_no": [0.3, 0.4],
                "prediction_1_for_yes": [0.7, 0.6],
                "label": ["yes", "yes"],
            }
        )

        automl = AutoML()
        automl._ml_task = BINARY_CLASSIFICATION
        automl._stacked_models = [_ModelMock("7_Optuna_LightGBM_T1", oof, pred)]

        X_stacked = automl.get_stacked_data(X.copy(), mode="predict")

        expected = "7_Optuna_LightGBM_T1_prediction_0_for_no_1_for_yes"
        self.assertIn(expected, X_stacked.columns.tolist())
        self.assertNotIn("7_Optuna_LightGBM_T1_prediction", X_stacked.columns.tolist())
        self.assertEqual(X_stacked[expected].tolist(), [0.7, 0.6])
