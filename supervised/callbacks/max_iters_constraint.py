import logging

log = logging.getLogger(__name__)

import numpy as np
import pandas as pd
from supervised.callbacks.callback import Callback
from supervised.metric import Metric


class MaxItersConstraint(Callback):
    def __init__(self, params):
        super(MaxItersConstraint, self).__init__(params)
        self.name = params.get("name", "max_iters_constraint")
        self.max_iters = params.get("max_iters", 10)

    def add_and_set_learner(self, learner):
        self.learner = learner

    def on_iteration_end(self, logs, predictions):
        # iters are computed starting from 0
        if logs.get("iter_cnt") + 1 >= self.max_iters:
            self.learner.stop_training = True
