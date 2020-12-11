import unittest
import tempfile
import json
import time
import numpy as np
import pandas as pd

from numpy.testing import assert_almost_equal
from supervised.callbacks.total_time_constraint import TotalTimeConstraint
from supervised.exceptions import AutoMLException


class TotalTimeConstraintTest(unittest.TestCase):
    def test_stop_on_first_learner(self):
        params = {
            "total_time_limit": 100,
            "total_time_start": time.time(),
            "expected_learners_cnt": 6000 + 1000 + 10,
        }
        callback = TotalTimeConstraint(params)
        callback.add_and_set_learner(learner={})
        callback.on_learner_train_start(logs=None)
        time.sleep(0.1)
        with self.assertRaises(AutoMLException) as context:
            callback.on_learner_train_end(logs=None)
        self.assertTrue("Stop training after first fold" in str(context.exception))

    def test_stop_on_not_first_learner(self):
        params = {
            "total_time_limit": 100,
            "total_time_start": time.time(),
            "expected_learners_cnt": 10,
        }
        callback = TotalTimeConstraint(params)
        callback.add_and_set_learner(learner={})
        callback.on_learner_train_start(logs=None)
        callback.on_learner_train_end(logs=None)
        with self.assertRaises(AutoMLException) as context:
            #
            # hardcoded change just for tests!
            callback.total_time_start = time.time() - 600 - 100 - 1
            #
            callback.add_and_set_learner(learner={})
            callback.on_learner_train_start(logs=None)
            callback.on_learner_train_end(logs=None)
        self.assertTrue("Force to stop" in str(context.exception))

    def test_dont_stop(self):
        params = {
            "total_time_limit": 100,
            "total_time_start": time.time(),
            "expected_learners_cnt": 10,
        }
        callback = TotalTimeConstraint(params)

        for i in range(10):
            callback.add_and_set_learner(learner={})
            callback.on_learner_train_start(logs=None)
            callback.on_learner_train_end(logs=None)
