import unittest
import tempfile
import json
import numpy as np
import pandas as pd

from iterative_learner_framework import IterativeLearner

class IterativeLearnerTest(unittest.TestCase):

    def test_create(self):
        train_params = {
            "preprocessing": {

            },
            "validation": {

            },
            "metrics": {

            },
            "model": {

            }
        }
        data = {
            'train': {
                'X': np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
                'y': np.array([0, 0, 1, 1])
            }
        }
        il = IterativeLearner(data, train_params)
