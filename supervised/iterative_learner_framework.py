
import numpy as np
import pandas as pd

from learner_framework import LearnerFramework

import logging
log = logging.getLogger(__name__)


from callbacks.early_stopping import EarlyStopping
from callbacks.time_constraint import TimeConstraint


from callbacks.callback_list import CallbackList

class IterativeLearnerError(Exception):
    pass


class IterativeLearner(LearnerFramework):

    def __init__(self, params, callbacks = []):
        LearnerFramework.__init__(self, data, train_params, callbacks)


        self.callbacks = CallbackList(callbacks)

        log.debug('IterativeLearner __init__')


    def train(self):
