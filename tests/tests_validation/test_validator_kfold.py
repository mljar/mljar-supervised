import unittest
import numpy as np
import pandas as pd
from supervised.validation.validator_kfold import KFoldValidator
from supervised.validation.validator_base import BaseValidatorException


class KFoldValidatorTest(unittest.TestCase):
    def test_create(self):
        data = {
            "train": {
                "X": pd.DataFrame(np.array([[0, 0], [0, 1], [1, 0], [1, 1]])),
                "y": pd.DataFrame(np.array([0, 0, 1, 1])),
            }
        }
        params = {"shuffle": False, "stratify": False, "k_folds": 2}
        vl = KFoldValidator(params, data)
        self.assertEqual(params["k_folds"], vl.get_n_splits())
        for train, validation in vl.split():
            X_train, y_train = train.get("X"), train.get("y")
            X_validation, y_validation = validation.get("X"), validation.get("y")

            self.assertEqual(X_train.shape[0], 2)
            self.assertEqual(y_train.shape[0], 2)
            self.assertEqual(X_validation.shape[0], 2)
            self.assertEqual(y_validation.shape[0], 2)

    def test_missing_target_values(self):
        # rows with missing target will be distributed equaly among folds
        data = {
            "train": {
                "X": pd.DataFrame(
                    np.array([[1, 0], [2, 1], [3, 0], [4, 1], [5, 1], [6, 1]])
                ),
                "y": pd.DataFrame(np.array(["a", "b", "a", "b", np.nan, np.nan])),
            }
        }
        params = {"shuffle": True, "stratify": True, "k_folds": 2}
        vl = KFoldValidator(params, data)
        self.assertEqual(params["k_folds"], vl.get_n_splits())
        for train, validation in vl.split():
            X_train, y_train = train.get("X"), train.get("y")
            X_validation, y_validation = validation.get("X"), validation.get("y")
            self.assertEqual(X_train.shape[0], 3)
            self.assertEqual(y_train.shape[0], 3)
            self.assertEqual(X_validation.shape[0], 3)
            self.assertEqual(y_validation.shape[0], 3)

    def test_create_with_target_as_labels(self):
        data = {
            "train": {
                "X": pd.DataFrame(np.array([[0, 0], [0, 1], [1, 0], [1, 1]])),
                "y": pd.DataFrame(np.array(["a", "b", "a", "b"])),
            }
        }
        params = {"shuffle": True, "stratify": True, "k_folds": 2}
        vl = KFoldValidator(params, data)
        self.assertEqual(params["k_folds"], vl.get_n_splits())
        for train, validation in vl.split():
            X_train, y_train = train.get("X"), train.get("y")
            X_validation, y_validation = validation.get("X"), validation.get("y")
            self.assertEqual(X_train.shape[0], 2)
            self.assertEqual(y_train.shape[0], 2)
            self.assertEqual(X_validation.shape[0], 2)
            self.assertEqual(y_validation.shape[0], 2)

    def test_missing_data(self):
        with self.assertRaises(BaseValidatorException) as context:
            data = {"train": {"X": np.array([[0, 0], [0, 1], [1, 0], [1, 1]])}}
            params = {"shuffle": True, "stratify": True, "k_folds": 2}
            vl = KFoldValidator(data, params)

        self.assertTrue("Missing" in str(context.exception))


if __name__ == "__main__":
    unittest.main()
