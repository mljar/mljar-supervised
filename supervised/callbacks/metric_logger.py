import logging

log = logging.getLogger(__name__)

import numpy as np

from supervised.callbacks.callback import Callback
from supervised.metric import Metric


class MetricLogger(Callback):
    def __init__(self, params):
        super(MetricLogger, self).__init__(params)
        self.name = params.get("name", "metric_logger")
        self.loss_values = {}
        self.metrics = []
        for metric_name in params.get("metric_names"):
            self.metrics += [Metric({"name": metric_name})]

    def add_and_set_learner(self, learner):
        self.loss_values[learner.uid] = {"train": {}, "validation": {}, "iters": []}
        for metric in self.metrics:
            self.loss_values[learner.uid]["train"][metric.name] = []
            self.loss_values[learner.uid]["validation"][metric.name] = []

        self.current_learner_uid = learner.uid

    def on_iteration_end(self, logs, predictions):

        for metric in self.metrics:
            train_loss = metric(
                predictions.get("y_train_true"), predictions.get("y_train_predicted")
            )
            validation_loss = metric(
                predictions.get("y_validation_true"),
                predictions.get("y_validation_predicted"),
            )
            self.loss_values[self.current_learner_uid]["train"][metric.name] += [
                train_loss
            ]
            self.loss_values[self.current_learner_uid]["validation"][metric.name] += [
                validation_loss
            ]
            # keep information about iter number only once :)
            if metric == self.metrics[0]:
                self.loss_values[self.current_learner_uid]["iters"] += [
                    logs.get("iter_cnt")
                ]
