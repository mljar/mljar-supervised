import unittest

from supervised.algorithms.factory import AlgorithmFactory
from supervised.algorithms.xgboost import XgbAlgorithm


class AlgorithmFactoryTest(unittest.TestCase):
    def test_fit(self):
        params = {
            "learner_type": "Xgboost",
            "objective": "binary:logistic",
            "eval_metric": "logloss",
        }
        learner = AlgorithmFactory.get_algorithm(params)
        self.assertEqual(
            learner.algorithm_short_name, XgbAlgorithm.algorithm_short_name
        )


if __name__ == "__main__":
    unittest.main()
