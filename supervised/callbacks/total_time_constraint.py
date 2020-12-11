import time
import logging
import numpy as np
from supervised.callbacks.callback import Callback
from supervised.exceptions import AutoMLException
from supervised.utils.config import LOG_LEVEL

log = logging.getLogger(__name__)
log.setLevel(LOG_LEVEL)


class TotalTimeConstraint(Callback):
    def __init__(self, params={}):
        super(TotalTimeConstraint, self).__init__(params)
        self.name = params.get("name", "total_time_constraint")
        self.total_time_limit = params.get("total_time_limit")
        self.total_time_start = params.get("total_time_start")
        self.expected_learners_cnt = params.get("expected_learners_cnt", 1)

    def on_learner_train_start(self, logs):
        self.train_start_time = time.time()

    def on_learner_train_end(self, logs):
        if (
            self.total_time_limit is not None
            and len(self.learners) == 1
            and self.expected_learners_cnt > 1
            # just check for the first learner
            # need to have more than 1 learner
            # otherwise it is a finish of the training
        ):
            one_fold_time = time.time() - self.train_start_time
            estimate_all_folds = one_fold_time * self.expected_learners_cnt

            total_elapsed_time = np.round(time.time() - self.total_time_start, 2)

            # we need to add time for the rest of learners (assuming that all folds training time is the same)
            estimate_elapsed_time = total_elapsed_time + one_fold_time * (
                self.expected_learners_cnt - 1
            )

            if estimate_elapsed_time >= self.total_time_limit:
                raise AutoMLException(
                    "Stop training after the first fold. "
                    f"Time needed to train on the first fold {np.round(one_fold_time)} seconds. "
                    "The time estimate for training on all folds is larger than total_time_limit."
                )
        if (
            self.total_time_limit is not None
            and len(self.learners) < self.expected_learners_cnt
            # dont stop for last learner, we are finishing anyway
        ):
            total_elapsed_time = np.round(time.time() - self.total_time_start, 2)

            if total_elapsed_time > self.total_time_limit + 600:
                # add 10 minutes of margin
                # margin is added because of unexpected time changes
                # if training on each fold will be the same
                # then the training will be stopped after first fold (above condition)
                raise AutoMLException(
                    "Force to stop the training. "
                    "Total time for AutoML training already exceeded."
                )

    def on_iteration_end(self, logs, predictions):

        total_elapsed_time = np.round(time.time() - self.total_time_start, 2)

        if self.total_time_limit is not None:
            log.debug(
                f"Total elapsed time {total_elapsed_time} seconds. "
                + f"Time left {np.round(self.total_time_limit - total_elapsed_time, 2)} seconds."
            )
            # not time left, stop now
            if total_elapsed_time >= self.total_time_limit:
                self.learner.stop_training = True
        else:
            log.debug(f"Total elapsed time {total_elapsed_time} seconds")
