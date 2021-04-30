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
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score


def logloss(y_true, y_predicted, sample_weight=None):
    epsilon = 1e-6
    y_predicted = sp.maximum(epsilon, y_predicted)
    y_predicted = sp.minimum(1 - epsilon, y_predicted)
    ll = log_loss(y_true, y_predicted, sample_weight=sample_weight)
    return ll


def rmse(y_true, y_predicted, sample_weight=None):
    val = mean_squared_error(y_true, y_predicted, sample_weight=sample_weight)
    return np.sqrt(val) if val > 0 else -np.Inf


def rmsle(y_true, y_predicted, sample_weight=None):
    val = mean_squared_log_error(y_true, y_predicted, sample_weight=sample_weight)
    return np.sqrt(val) if val > 0 else -np.Inf


def negative_auc(y_true, y_predicted, sample_weight=None):
    val = roc_auc_score(y_true, y_predicted, sample_weight=sample_weight)
    return -1.0 * val


def negative_r2(y_true, y_predicted, sample_weight=None):
    val = r2_score(y_true, y_predicted, sample_weight=sample_weight)
    return -1.0 * val


def negative_f1(y_true, y_predicted, sample_weight=None):

    if isinstance(y_true, pd.DataFrame):
        y_true = np.array(y_true)
    if isinstance(y_predicted, pd.DataFrame):
        y_predicted = np.array(y_predicted)

    if len(y_predicted.shape) == 2 and y_predicted.shape[1] == 1:
        y_predicted = y_predicted.ravel()

    average = None
    if len(y_predicted.shape) == 1:
        y_predicted = (y_predicted > 0.5).astype(int)
        average = "binary"
    else:
        y_predicted = np.argmax(y_predicted, axis=1)
        average = "micro"

    val = f1_score(y_true, y_predicted, sample_weight=sample_weight, average=average)

    return -val


def negative_accuracy(y_true, y_predicted, sample_weight=None):

    if isinstance(y_true, pd.DataFrame):
        y_true = np.array(y_true)
    if isinstance(y_predicted, pd.DataFrame):
        y_predicted = np.array(y_predicted)

    if len(y_predicted.shape) == 2 and y_predicted.shape[1] == 1:
        y_predicted = y_predicted.ravel()

    if len(y_predicted.shape) == 1:
        y_predicted = (y_predicted > 0.5).astype(int)
    else:
        y_predicted = np.argmax(y_predicted, axis=1)

    val = accuracy_score(y_true, y_predicted, sample_weight=sample_weight)

    return -val


def negative_average_precision(y_true, y_predicted, sample_weight=None):

    if isinstance(y_true, pd.DataFrame):
        y_true = np.array(y_true)
    if isinstance(y_predicted, pd.DataFrame):
        y_predicted = np.array(y_predicted)

    val = average_precision_score(y_true, y_predicted, sample_weight=sample_weight)

    return -val


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
    return -np.corrcoef(y_true, y_predicted)[0, 1]


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
    return "r2", -r2_score(target, preds, sample_weight=weight)


def xgboost_eval_metric_spearman(preds, dtrain):
    # Xgboost needs to minimize eval_metric
    target = dtrain.get_label()
    return "spearman", negative_spearman(target, preds)


def xgboost_eval_metric_pearson(preds, dtrain):
    # Xgboost needs to minimize eval_metric
    target = dtrain.get_label()
    return "pearson", negative_pearson(target, preds)


def xgboost_eval_metric_f1(preds, dtrain):
    # Xgboost needs to minimize eval_metric
    target = dtrain.get_label()
    weight = dtrain.get_weight()
    if len(weight) == 0:
        weight = None
    return "f1", negative_f1(target, preds, weight)


def xgboost_eval_metric_average_precision(preds, dtrain):
    # Xgboost needs to minimize eval_metric
    target = dtrain.get_label()
    weight = dtrain.get_weight()
    if len(weight) == 0:
        weight = None
    return "average_precision", negative_average_precision(target, preds, weight)


def xgboost_eval_metric_accuracy(preds, dtrain):
    # Xgboost needs to minimize eval_metric
    target = dtrain.get_label()
    weight = dtrain.get_weight()
    if len(weight) == 0:
        weight = None
    return "accuracy", negative_accuracy(target, preds, weight)


def xgboost_eval_metric_mse(preds, dtrain):
    # Xgboost needs to minimize eval_metric
    target = dtrain.get_label()
    weight = dtrain.get_weight()
    if len(weight) == 0:
        weight = None
    return "mse", mean_squared_error(target, preds, sample_weight=weight)


def lightgbm_eval_metric_r2(preds, dtrain):
    target = dtrain.get_label()
    weight = dtrain.get_weight()
    return "r2", r2_score(target, preds, sample_weight=weight), True


def lightgbm_eval_metric_spearman(preds, dtrain):
    target = dtrain.get_label()
    return "spearman", -negative_spearman(target, preds), True


def lightgbm_eval_metric_pearson(preds, dtrain):
    target = dtrain.get_label()
    return "pearson", -negative_pearson(target, preds), True


def lightgbm_eval_metric_f1(preds, dtrain):
    target = dtrain.get_label()
    weight = dtrain.get_weight()

    unique_targets = np.unique(target)
    if len(unique_targets) > 2:
        cols = len(unique_targets)
        rows = int(preds.shape[0] / len(unique_targets))
        preds = np.reshape(preds, (rows, cols), order="F")

    return "f1", -negative_f1(target, preds, weight), True


def lightgbm_eval_metric_average_precision(preds, dtrain):
    target = dtrain.get_label()
    weight = dtrain.get_weight()

    return "average_precision", -negative_average_precision(target, preds, weight), True


def lightgbm_eval_metric_accuracy(preds, dtrain):
    target = dtrain.get_label()
    weight = dtrain.get_weight()

    unique_targets = np.unique(target)
    if len(unique_targets) > 2:
        cols = len(unique_targets)
        rows = int(preds.shape[0] / len(unique_targets))
        preds = np.reshape(preds, (rows, cols), order="F")

    return "accuracy", -negative_accuracy(target, preds, weight), True


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


class CatBoostEvalMetricAveragePrecision(object):
    def get_final_error(self, error, weight):
        return error

    def is_max_optimal(self):
        return True

    def evaluate(self, approxes, target, weight):
        assert len(approxes) == 1
        assert len(target) == len(approxes[0])

        preds = np.array(approxes[0])
        target = np.array(target)
        if weight is not None:
            weight = np.array(weight)

        return -negative_average_precision(target, preds, weight), 0


class CatBoostEvalMetricMSE(object):
    def get_final_error(self, error, weight):
        return error

    def is_max_optimal(self):
        return False

    def evaluate(self, approxes, target, weight):
        assert len(approxes) == 1
        assert len(target) == len(approxes[0])

        preds = np.array(approxes[0])
        target = np.array(target)
        if weight is not None:
            weight = np.array(weight)

        return mean_squared_error(target, preds, sample_weight=weight), 0


class UserDefinedEvalMetric:
    # should always minimize
    eval_metric = mean_squared_error  # set the default

    def set_metric(self, feval):
        UserDefinedEvalMetric.eval_metric = feval

    def __call__(self, y_true, y_predicted, sample_weight=None):
        return UserDefinedEvalMetric.eval_metric(y_true, y_predicted, sample_weight)


def xgboost_eval_metric_user_defined(preds, dtrain):
    target = dtrain.get_label()
    weight = dtrain.get_weight()
    if len(weight) == 0:
        weight = None
    metric = UserDefinedEvalMetric()
    return "user_defined_metric", metric(target, preds, sample_weight=weight)


def lightgbm_eval_metric_user_defined(preds, dtrain):
    target = dtrain.get_label()
    weight = dtrain.get_weight()
    metric = UserDefinedEvalMetric()
    return "user_defined_metric", metric(target, preds, sample_weight=weight), False


class CatBoostEvalMetricUserDefined(object):
    def get_final_error(self, error, weight):
        return error

    def is_max_optimal(self):
        return False

    def evaluate(self, approxes, target, weight):
        assert len(approxes) == 1
        assert len(target) == len(approxes[0])

        preds = np.array(approxes[0])
        target = np.array(target)
        if weight is not None:
            weight = np.array(weight)

        metric = UserDefinedEvalMetric()
        return metric(target, preds, sample_weight=weight), 0


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
            "spearman",  # negative
            "pearson",  # negative
            "f1",  # negative
            "average_precision",  # negative
            "accuracy",  # negative
            "user_defined_metric",
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
        elif self.name == "f1":
            self.metric = negative_f1
        elif self.name == "average_precision":
            self.metric = negative_average_precision
        elif self.name == "accuracy":
            self.metric = negative_accuracy
        elif self.name == "user_defined_metric":
            self.metric = UserDefinedEvalMetric.eval_metric
        # elif self.name == "rmsle": # need to update target preprocessing
        #    self.metric = rmsle     # to assure that target is not negative ...
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
        return self.name in [
            "auc",
            "r2",
            "spearman",
            "pearson",
            "f1",
            "average_precision",
            "accuracy",
        ]

    @staticmethod
    def optimize_negative(metric_name):
        return metric_name in [
            "auc",
            "r2",
            "spearman",
            "pearson",
            "f1",
            "average_precision",
            "accuracy",
        ]
