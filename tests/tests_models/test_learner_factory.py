import unittest
import tempfile
import json
import numpy as np
import pandas as pd

from supervised.models.learner_factory import LearnerFactory

from supervised.models.learner_xgboost import XgbLearner

class LearnerFactoryTest(unittest.TestCase):
    def test_fit(self):
        params = {
            "learner_type": "Xgboost",
            "objective": "binary:logistic",
            "eval_metric": "logloss",
        }
        learner = LearnerFactory.get_learner(params)
        self.assertEqual(learner.algorithm_short_name, XgbLearner.algorithm_short_name)

if __name__ == "__main__":
    unittest.main()
