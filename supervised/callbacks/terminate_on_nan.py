import logging

log = logging.getLogger(__name__)

import numpy as np
from supervised.callbacks.callback import Callback


class TerminateOnNan(Callback):
    def __init__(self, learner, params):
        super(TerminateOnNan, self).__init__(learner, params)
        self.metric = Metric(params.get("metric_name"))

    def on_iteration_end(self, iter_cnt, data):
        loss_train = self.metric(
            data.get("y_train_true"), data.get("y_train_predicted")
        )
        loss_validation = self.metric(
            data.get("y_validation_true"), data.get("y_validation_predicted")
        )

        for loss in [loss_train, loss_validation]:
            if np.isnan(loss) or np.isinf(loss) or np.isneginf(loss):
                self.learner.stop_training = True
                log.info("Terminating learning, invalid loss value")
