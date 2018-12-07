import logging

log = logging.getLogger(__name__)

import numpy as np

from supervised.callbacks.callback import Callback
from metric import Metric


class EarlyStopping(Callback):
    def __init__(self, params):
        super(EarlyStopping, self).__init__(params)
        self.metric = Metric(params.get("metric"))
        self.max_no_improvement_cnt = params.get("max_no_improvement_cnt", 5)

        self.keep_best_model = params.get("keep_best_model", True)
        self.best_loss = {}
        self.loss_values = {}
        self.best_models = {}
        # path to best model local copy, only used if cannot deep copy
        self.best_model_paths = {}

    def add_and_set_learner(self, learner):
        self.learners += [learner]
        self.learner = learner
        self.best_loss[learner.uid] = self.metric.worst_value()
        self.loss_values[learner.uid] = {"values": [], "iters": []}
        self.best_models[learner.uid] = None
        self.best_model_paths[learner.uid] = None

    def on_learner_train_start(self, logs):
        self.no_improvement_cnt = 0

    def on_iteration_end(self, logs, predictions):
        loss = self.metric(
            predictions.get("y_validation_true"),
            predictions.get("y_validation_predicted"),
        )
        self.loss_values[self.learner.uid]["values"] += [loss]
        self.loss_values[self.learner.uid]["iters"] += [logs.get("iter_cnt")]

        if self.metric.improvement(
            previous=self.best_loss[self.learner.uid], current=loss
        ):
            self.no_improvement_cnt = 0
            self.best_loss[self.learner.uid] = loss
            self.best_models[self.learner.uid] = self.learner.copy()
            # if local copy is not available, save model and keep path
            if self.best_models[self.learner.uid] is None:
                self.best_model_paths[self.learner.uid] = self.learner.save()
        else:
            self.no_improvement_cnt += 1

        if self.no_improvement_cnt > self.max_no_improvement_cnt:
            self.learner.stop_training = True

        log.debug(
            "EarlyStopping.on_teration_end, loss: {0}, "
            "no improvement cnt {1}".format(loss, self.no_improvement_cnt)
        )
