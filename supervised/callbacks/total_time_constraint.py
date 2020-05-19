import time
import logging
import numpy as np
from supervised.callbacks.callback import Callback

from supervised.utils.config import LOG_LEVEL

log = logging.getLogger(__name__)
log.setLevel(LOG_LEVEL)


class TotalTimeConstraint(Callback):
    def __init__(self, params={}):
        super(TotalTimeConstraint, self).__init__(params)
        self.name = params.get("name", "total_time_constraint")
        self.total_time_limit = params.get("total_time_limit")
        self.total_time_start = params.get("total_time_start")

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
