import unittest
import tempfile
import json
import numpy as np
import pandas as pd

from learner_xgboost import XgbLearner

class XgboostLearnerTest(unittest.TestCase):

    def test_fit(self):
        params = {}
        xgb = XgbLearner(params)
        data = {
            'train': {
                'X': np.array([[0,1,2,3], [0,1,2,3]]),
                'y': np.array([0,1])
            }
        }
        xgb.fit(data['train'])

        # xgb.copy() # copy.deepcopy(clf) # error
