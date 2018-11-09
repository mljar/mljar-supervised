import unittest
import tempfile
import json
import numpy as np
import pandas as pd

from iterative_learner_framework import IterativeLearner

class IterativeLearnerTest(unittest.TestCase):

    def test_create(self):
        train_params = {
            'preprocessing': {},
            'validation': {
                'validation_type': 'split',
                'train_ratio': 0.5,
                'shuffle': True
            },
            'learner': {
                'learner_type': 'xgb',
                'max_iters': 4,
                'silent':1,
                'max_depth': 1
            }
        }
        data = {
            'train': {
                'X': np.array([[0, 0], [0, 1], [0.1, 0], [0.2, 1], [1, 0], [1, 1]]),
                'y': np.array([0, 0, 0, 0, 1, 1])
            }
        }
        il = IterativeLearner(train_params, callbacks = [])
        il.train(data)
