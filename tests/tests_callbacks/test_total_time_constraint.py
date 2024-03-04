import time
import unittest

from supervised.callbacks.total_time_constraint import TotalTimeConstraint
from supervised.exceptions import NotTrainedException


class TotalTimeConstraintTest(unittest.TestCase):
    def test_stop_on_first_learner(self):
        params = {
            "total_time_limit": 100,
            "total_time_start": time.time(),
            "expected_learners_cnt": 1001,
        }
        callback = TotalTimeConstraint(params)
        callback.add_and_set_learner(learner={})
        callback.on_learner_train_start(logs=None)
        time.sleep(0.1)
        with self.assertRaises(NotTrainedException) as context:
            callback.on_learner_train_end(logs=None)
        self.assertTrue("Stop training after the first fold" in str(context.exception))

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
        with self.assertRaises(NotTrainedException) as context:
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
