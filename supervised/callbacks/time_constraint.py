import time
import logging
from supervised.callbacks.callback import Callback

log = logging.getLogger(__name__)


class TimeConstraint(Callback):
    def __init__(self, params={}):
        super(TimeConstraint, self).__init__(params)
        self.name = params.get("name", "time_constraint")
        self.train_time_limit = params.get("train_seconds_time_limit", 60)  # in seconds
        self.last_iteration_time = 0

    def on_learner_train_start(self, logs):
        self.train_start_time = time.time()

    def on_iteration_start(self, logs):
        self.iter_start_time = time.time()

    def on_iteration_end(self, logs, predictions):
        log.debug(
            "Iteration took {0} seconds, total training time {1} seconds".format(
                time.time() - self.iter_start_time, time.time() - self.train_start_time
            )
        )
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
