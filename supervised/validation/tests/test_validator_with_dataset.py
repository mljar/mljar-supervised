import unittest
import numpy as np
from validator_with_dataset import WithDatasetValidator
from validator_with_dataset import WithDatasetValidatorException


class WithDatasetValidatorTest(unittest.TestCase):
    def test_create(self):
        data = {
            "train": {
                "X": np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
                "y": np.array([0, 0, 1, 1]),
            },
            "validation": {
                "X": np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
                "y": np.array([0, 0, 1, 1]),
            },
        }
        vl = WithDatasetValidator(data, {})
        self.assertEqual(1, vl.get_n_splits())
        cnt = 0
        for X_train, y_train, X_validation, y_validation in vl.split():
            self.assertEqual(X_train.shape[0], 4)
            self.assertEqual(y_train.shape[0], 4)
            self.assertEqual(X_validation.shape[0], 4)
            self.assertEqual(y_validation.shape[0], 4)
            cnt += 1
        self.assertEqual(cnt, 1)

    def test_missing_data(self):
        with self.assertRaises(WithDatasetValidatorException) as context:
            data = {
                "train": {
                    "X": np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
                    "y": np.array([0, 0, 1, 1]),
                },
                "validation": {"X": np.array([[0, 0], [0, 1], [1, 0], [1, 1]])},
            }
            vl = WithDatasetValidator(data, {})

        self.assertTrue("Missing" in str(context.exception))
