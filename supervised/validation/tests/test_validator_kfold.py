import unittest
import numpy as np
from validator_kfold import KFoldValidator
from validator_base import BaseValidatorException


class KFoldValidatorTest(unittest.TestCase):
    def test_create(self):
        data = {
            "train": {
                "X": np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
                "y": np.array([0, 0, 1, 1]),
            }
        }
        params = {"shuffle": False, "stratify": False, "k_folds": 2}
        vl = KFoldValidator(data, params)
        self.assertEqual(params["k_folds"], vl.get_n_splits())
        for X_train, y_train, X_validation, y_validation in vl.split():
            self.assertEqual(X_train.shape[0], 2)
            self.assertEqual(y_train.shape[0], 2)
            self.assertEqual(X_validation.shape[0], 2)
            self.assertEqual(y_validation.shape[0], 2)

    def test_create_with_target_as_labels(self):
        data = {
            "train": {
                "X": np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
                "y": np.array(["a", "b", "a", "b"]),
            }
        }
        params = {"shuffle": True, "stratify": True, "k_folds": 2}
        vl = KFoldValidator(data, params)
        self.assertEqual(params["k_folds"], vl.get_n_splits())
        for X_train, y_train, X_validation, y_validation in vl.split():
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
