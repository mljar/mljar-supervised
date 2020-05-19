import time
import logging
import numpy as np
from supervised.callbacks.callback import Callback

from supervised.utils.config import LOG_LEVEL

log = logging.getLogger(__name__)
log.setLevel(LOG_LEVEL)


class LearnerTimeConstraint(Callback):
    def __init__(self, params={}):
        super(LearnerTimeConstraint, self).__init__(params)
        self.name = params.get("name", "learner_time_constraint")
        self.min_steps = params.get("min_steps")
        self.learner_time_limit = params.get("learner_time_limit")  # in seconds
        self.iterations_count = 0

    def on_learner_train_start(self, logs):
        self.train_start_time = time.time()
        self.iterations_count = 0

    def on_iteration_start(self, logs):
        self.iter_start_time = time.time()

    def on_iteration_end(self, logs, predictions):
        self.iterations_count += 1
        iteration_elapsed_time = np.round(time.time() - self.iter_start_time, 2)
        learner_elapsed_time = np.round(time.time() - self.train_start_time, 2)
        log.debug(
            "Iteration {0} took {1} seconds, learner training time {2} seconds".format(
                self.iterations_count, iteration_elapsed_time, learner_elapsed_time
            )
        )

        if self.min_steps is not None:
            if self.iterations_count < self.min_steps:
                # self.learner.stop_training = False
                # return before checking other conditions
                return

        if self.learner_time_limit is not None:
            if learner_elapsed_time >= self.learner_time_limit:
                self.learner.stop_training = True
                log.info("Terminating learning, time limit reached")
