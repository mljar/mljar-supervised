import unittest
import numpy as np
from validator_split import SplitValidator
from validator_split import SplitValidatorException


class SplitValidatorTest(unittest.TestCase):
    def test_create(self):
        data = {
            "train": {
                "X": np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
                "y": np.array([0, 0, 1, 1]),
            }
        }
        params = {"shuffle": False, "stratify": False, "train_ratio": 0.75}
        vl = SplitValidator(data, params)
        self.assertEqual(1, vl.get_n_splits())
        cnt = 0
        for X_train, y_train, X_validation, y_validation in vl.split():
            self.assertEqual(X_train.shape[0], 3)
            self.assertEqual(y_train.shape[0], 3)
            self.assertEqual(X_validation.shape[0], 1)
            self.assertEqual(X_validation.shape[1], 2)
            self.assertEqual(y_validation.shape[0], 1)
            cnt += 1
        self.assertEqual(cnt, 1)

    def wrong_split_value(self, split_value):
        with self.assertRaises(ValueError) as context:
            data = {
                "train": {
                    "X": np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
                    "y": np.array([0, 0, 1, 1]),
                }
            }
            params = {"shuffle": True, "stratify": True, "train_ratio": split_value}
            vl = SplitValidator(data, params)
            X_train, y_train, X_validation, y_validation = vl.split()
        self.assertTrue("should be" in str(context.exception))

    def test_wrong_split_values(self):
        for i in [0.1, 0.9, 1.1, -0.1]:
            self.wrong_split_value(i)
