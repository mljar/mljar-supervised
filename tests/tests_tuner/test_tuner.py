import os
import unittest
from supervised.tuner.mljar_tuner import MljarTuner


class TunerTest(unittest.TestCase):
    def test_key_params(self):
        params1 = {
            "preprocessing": {
                "p1": 1,
                "p2": 2
            },
            "learner": {
                "p1": 1,
                "p2": 2
            }
        }
        params2 = {
            "preprocessing": {
                "p1": 1,
                "p2": 2
            },
            "learner": {
                "p2": 2,
                "p1": 1
            }
        }
        key1 = MljarTuner.get_params_key(params1)
        key2 = MljarTuner.get_params_key(params2)
        self.assertEqual(key1, key2)
        
