
import numpy as np
import pandas as pd

from learner_framework import LearnerFramework

import logging
log = logging.getLogger(__name__)

class IterativeLearnerError(Exception):
    pass


class IterativeLearner(LearnerFramework):

    def __init__(self, data, train_params, callbacks = None):
        LearnerFramework.__init__(self, data, train_params, callbacks)
        log.debug('IterativeLearner __init__')
