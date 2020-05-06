import time
import logging
from supervised.callbacks.callback import Callback

from supervised.utils.config import LOG_LEVEL

log = logging.getLogger(__name__)
log.setLevel(LOG_LEVEL)


class TimeConstraint(Callback):
    def __init__(self, params={}):
        super(TimeConstraint, self).__init__(params)
        self.name = params.get("name", "time_constraint")
        self.train_time_limit = params.get("train_seconds_time_limit", 60)  # in seconds
        self.min_steps = params.get("min_steps")
        self.total_time_limit = params.get("total_time_limit")
        self.total_time_start = params.get("total_time_start")
        
        self.last_iteration_time = 0
        self.iterations_count = 0

    def on_learner_train_start(self, logs):
        self.train_start_time = time.time()
        self.iterations_count = 0

    def on_iteration_start(self, logs):
        self.iter_start_time = time.time()

    def on_iteration_end(self, logs, predictions):
        self.iterations_count += 1
        log.debug(
            "Iteration {0} took {1} seconds, total training time {1} seconds".format(
                self.iterations_count,
                time.time() - self.iter_start_time,
                time.time() - self.train_start_time,
            )
        )

        if self.total_time_limit is not None:
            # not time left, stop now
            if time.time() - self.total_time_start > self.total_time_limit:
                self.learner.stop_training = True
                return

        if time.time() - self.train_start_time > self.train_time_limit:
            self.learner.stop_training = True
            log.info("Terminating learning, time limit reached")
        if (
            time.time() - self.train_start_time + self.last_iteration_time
            > self.train_time_limit
        ):
            self.learner.stop_training = True
            log.info("Terminating learning, time limit will be exceeded")

        self.last_iteration_time = time.time() - self.iter_start_time

        if self.min_steps is not None:
            if self.iterations_count < self.min_steps:
                self.learner.stop_training = False
