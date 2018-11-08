import logging
log = logging.getLogger(__name__)

import numpy as np
import scipy as sp
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score

class MetricException(Exception):
    def __init__(self, message):
        Exception.__init__(self, message)
        log.error(message)

def logloss(y_true, y_predicted):
    epsilon = 1e-15
    y_predicted = sp.maximum(epsilon, y_predicted)
    y_predicted = sp.minimum(1-epsilon, y_predicted)
    return log_loss(y_true, y_predicted)

def rmse(y_true, y_predicted):
    val = mean_squared_error(y_true, y_predicted)
    return np.sqrt(val) if val > 0 else -np.Inf

class Metric(object):

    def __init__(self, params):
        self.params = params
        self.metric_name = self.params.get('metric_name')
        if self.metric_name is None:
            raise MetricException('Metric not defined')
        self.minimize_direction = self.metric_name in ['logloss', 'rmse',
                                                        'mae', 'ce', 'mse']
        if self.metric_name == 'logloss':
            self.metric = logloss
        elif self.metric_name == 'auc':
            self.metric = roc_auc_score
        elif self.metric_name == 'acc':
            self.metric = accuracy_score
        elif self.metric_name == 'rmse':
            self.metric = rmse
        elif self.metric_name == 'mse':
            self.metric = mean_squared_error
        elif self.metric_name == 'mae':
            self.metric = mean_absolute_error
        else:
            raise MetricException('Unknown metric {0}'.format(self.metric_name))

    def __call__(self, y_true, y_predicted):
        return self.metric(y_true, y_predicted)

    def improvement(self, previous, current):
        if self.minimize_direction:
            return current < previous
        return current > previous
