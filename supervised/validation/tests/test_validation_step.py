import unittest
import numpy as np
from validation_step import ValidationStep
from validation_step import ValidationStepException


class ValidationStepTest(unittest.TestCase):
    def test_create(self):
        data = {
            "train": {
                "X": np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
                "y": np.array([0, 0, 1, 1]),
            }
        }
        params = {
            "validator_type": "split",
            "shuffle": False,
            "stratify": False,
            "train_ratio": 0.5,
        }
        vl = ValidationStep(data, params)
        self.assertEqual(1, vl.get_n_splits())
        for X_train, y_train, X_validation, y_validation in vl.split():
            self.assertEqual(X_train.shape[0], 2)
            self.assertEqual(y_train.shape[0], 2)
            self.assertEqual(X_validation.shape[0], 2)
            self.assertEqual(y_validation.shape[0], 2)

    def test_wrong_validator_type(self):
        with self.assertRaises(ValidationStepException) as context:
            data = {}
            params = {"validator_type": "no_such_validator"}
            vl = ValidationStep(data, params)

        self.assertTrue("Unknown" in str(context.exception))
