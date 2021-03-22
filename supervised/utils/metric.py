import logging

log = logging.getLogger(__name__)

import numpy as np
import pandas as pd
import scipy as sp
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error



def logloss(y_true, y_predicted, sample_weight=None):
    epsilon = 1e-6
    y_predicted = sp.maximum(epsilon, y_predicted)
    y_predicted = sp.minimum(1 - epsilon, y_predicted)
    ll = log_loss(y_true, y_predicted, sample_weight=sample_weight)
    return ll


def rmse(y_true, y_predicted, sample_weight=None):
    val = mean_squared_error(y_true, y_predicted, sample_weight=sample_weight)
    return np.sqrt(val) if val > 0 else -np.Inf


def negative_auc(y_true, y_predicted, sample_weight=None):
    val = roc_auc_score(y_true, y_predicted, sample_weight=sample_weight)
    return -1.0 * val


def negative_r2(y_true, y_predicted, sample_weight=None):
    val = r2_score(y_true, y_predicted, sample_weight=sample_weight)
    return -1.0 * val

def negative_spearman(y_true, y_predicted, sample_weight=None):
    # sample weight is ignored
    c, _ = sp.stats.spearmanr(y_true, y_predicted)
    return -c

def negative_pearson(y_true, y_predicted, sample_weight=None):
    # sample weight is ignored
    if isinstance(y_true, pd.DataFrame):
        y_true = np.array(y_true).ravel()
    if isinstance(y_predicted, pd.DataFrame):
        y_predicted = np.array(y_predicted).ravel()
    return -np.corrcoef(y_true, y_predicted)[0,1]




class MetricException(Exception):
    def __init__(self, message):
        Exception.__init__(self, message)
        log.error(message)


def xgboost_eval_metric_r2(preds, dtrain):
    # Xgboost needs to minimize eval_metric
    target = dtrain.get_label()    
    weight = dtrain.get_weight()
    if len(weight) == 0:
        weight = None 
    return 'r2', -r2_score(target, preds, sample_weight=weight)

def xgboost_eval_metric_spearman(preds, dtrain):
    # Xgboost needs to minimize eval_metric
    target = dtrain.get_label()    
    return 'spearman', negative_spearman(target, preds)

def xgboost_eval_metric_pearson(preds, dtrain):
    # Xgboost needs to minimize eval_metric
    target = dtrain.get_label()    
    return 'pearson', negative_pearson(target, preds)

def lightgbm_eval_metric_r2(preds, dtrain):
    target = dtrain.get_label()
    weight = dtrain.get_weight()
    return 'r2', r2_score(target, preds, sample_weight=weight), True

def lightgbm_eval_metric_spearman(preds, dtrain):
    target = dtrain.get_label()
    return 'spearman', -negative_spearman(target, preds), True

def lightgbm_eval_metric_pearson(preds, dtrain):
    target = dtrain.get_label()
    return 'pearson', -negative_pearson(target, preds), True


class CatBoostEvalMetricSpearman(object):
    def get_final_error(self, error, weight):
        return error

    def is_max_optimal(self):
        return True

    def evaluate(self, approxes, target, weight):
        assert len(approxes) == 1
        assert len(target) == len(approxes[0])

        preds = np.array(approxes[0])
        target = np.array(target)

        return -negative_spearman(target, preds), 0

class CatBoostEvalMetricPearson(object):
    def get_final_error(self, error, weight):
        return error

    def is_max_optimal(self):
        return True

    def evaluate(self, approxes, target, weight):
        assert len(approxes) == 1
        assert len(target) == len(approxes[0])

        preds = np.array(approxes[0])
        target = np.array(target)

        return -negative_pearson(target, preds), 0


class Metric(object):
    def __init__(self, params):
        if params is None:
            raise MetricException("Metric params not defined")
        self.params = params
        self.name = self.params.get("name")
        if self.name is None:
            raise MetricException("Metric name not defined")

        self.minimize_direction = self.name in [
            "logloss",
            "auc",  # negative auc
            "rmse",
            "mae",
            "mse",
            "r2",  # negative r2
            "mape",
            "spearman", # negative
            "pearson" # negative
        ]
        if self.name == "logloss":
            self.metric = logloss
        elif self.name == "auc":
            self.metric = negative_auc
        elif self.name == "acc":
            self.metric = accuracy_score
        elif self.name == "rmse":
            self.metric = rmse
        elif self.name == "mse":
            self.metric = mean_squared_error
        elif self.name == "mae":
            self.metric = mean_absolute_error
        elif self.name == "r2":
            self.metric = negative_r2
        elif self.name == "mape":
            self.metric = mean_absolute_percentage_error
        elif self.name == "spearman":
            self.metric = negative_spearman
        elif self.name == "pearson":
            self.metric = negative_pearson
        else:
            raise MetricException(f"Unknown metric '{self.name}'")

    def __call__(self, y_true, y_predicted, sample_weight=None):
        return self.metric(y_true, y_predicted, sample_weight=sample_weight)

    def improvement(self, previous, current):
        if self.minimize_direction:
            return current < previous
        return current > previous

    def get_maximum(self):
        if self.minimize_direction:
            return 10e12
        else:
            return -10e12

    def worst_value(self):
        if self.minimize_direction:
            return np.Inf
        return -np.Inf

    def get_minimize_direction(self):
        return self.minimize_direction

    def is_negative(self):
        return self.name in ["auc", "r2", "spearman", "pearson"]

    @staticmethod
    def optimize_negative(metric_name):
        return metric_name in ["auc", "r2", "spearman", "pearson"]
