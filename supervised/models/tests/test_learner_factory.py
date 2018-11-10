import unittest
import tempfile
import json
import numpy as np
import pandas as pd

from supervised.models.learner_factory import LearnerFactory

class LearnerFactoryTest(unittest.TestCase):

    def test_fit(self):
        params = {
            'learner_type': 'xgb',
            'objective': 'binary:logistic',
            'eval_metric': 'logloss'
        }
        learner = LearnerFactory.get_learner(params)
