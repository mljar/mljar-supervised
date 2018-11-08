import logging
log = logging.getLogger(__name__)

import numpy as np

from callback import Callback

class EarlyStopping(Callback):

    def __init__(self, learner, params):
        super(EarlyStopping, self).__init__(learner, params, data)
        self.metric = Metric(params.get('metric_name'))
        self.max_no_improvement_cnt = params.get('max_no_improvement_cnt', 5)
        self.keep_best_model = params.get('keep_best_model', True)
        self.best_model = None
        # path to best model local copy, only used if cannot deep copy
        self.best_model_path = None
        self.loss_values = []
        self.iter_cnts = []

    def on_iteration_end(self, iter_cnt, data):
        loss = self.metric(data.get('y_validation_true'),
                            data.get('y_validation_predicted'))
        self.loss_values += [loss]
        self.iter_cnts += [iter_cnt]

        if self.metric.improvement(score_previous = self.loss_values[-2],
                                    score_current = self.loss_values[-1]):
            self.best_model = self.learner.copy()
            # if local copy is not available, save model and keep path
            if self.best_model is None:
                self.best_model_path = self.learner.save()
        else:
            no_improvement_cnt += 1

        if no_improvement_cnt > self.max_no_improvement_cnt:
            self.learner.stop_training = True
