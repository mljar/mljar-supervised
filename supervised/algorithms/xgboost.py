import logging
import copy
import numpy as np
import pandas as pd
import os
import time
from inspect import signature
import xgboost as xgb

from supervised.algorithms.algorithm import BaseAlgorithm
from supervised.algorithms.registry import AlgorithmsRegistry
from supervised.algorithms.registry import (
    BINARY_CLASSIFICATION,
    MULTICLASS_CLASSIFICATION,
    REGRESSION,
)
from supervised.utils.metric import (
    xgboost_eval_metric_r2,
    xgboost_eval_metric_spearman,
    xgboost_eval_metric_pearson,
    xgboost_eval_metric_f1,
    xgboost_eval_metric_average_precision,
    xgboost_eval_metric_accuracy,
    xgboost_eval_metric_mse,
    xgboost_eval_metric_user_defined,
)
from supervised.utils.config import LOG_LEVEL

logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)

import tempfile


class XgbAlgorithmException(Exception):
    def __init__(self, message):
        super(XgbAlgorithmException, self).__init__(message)
        logger.error(message)


def time_constraint(env):
    # print("time constraint")
    pass


def xgboost_eval_metric(ml_task, automl_eval_metric):
    # the mapping is almost the same
    eval_metric_name = automl_eval_metric
    if ml_task == MULTICLASS_CLASSIFICATION:
        if automl_eval_metric == "logloss":
            eval_metric_name = "mlogloss"
    return eval_metric_name


def xgboost_objective(ml_task, automl_eval_metric):
    objective = "reg:squarederror"
    if ml_task == BINARY_CLASSIFICATION:
        objective = "binary:logistic"
    elif ml_task == MULTICLASS_CLASSIFICATION:
        objective = "multi:softprob"
    else:  # ml_task == REGRESSION
        objective = "reg:squarederror"
    return objective


class XgbAlgorithm(BaseAlgorithm):
    """
    This is a wrapper over xgboost algorithm.
    """

    algorithm_name = "Extreme Gradient Boosting"
    algorithm_short_name = "Xgboost"

    def __init__(self, params):
        super(XgbAlgorithm, self).__init__(params)
        self.library_version = xgb.__version__

        self.explain_level = params.get("explain_level", 0)
        self.boosting_rounds = additional.get("max_rounds", 10000)
        self.max_iters = 1
        self.early_stopping_rounds = additional.get("early_stopping_rounds", 50)
        self.learner_params = {
            "tree_method": "hist",
            "booster": "gbtree",
            "objective": self.params.get("objective"),
            "eval_metric": self.params.get("eval_metric"),
            "eta": self.params.get("eta", 0.01),
            "max_depth": self.params.get("max_depth", 1),
            "min_child_weight": self.params.get("min_child_weight", 1),
            "subsample": self.params.get("subsample", 0.8),
            "colsample_bytree": self.params.get("colsample_bytree", 0.8),
            "n_jobs": self.params.get("n_jobs", -1),
            # "silent": self.params.get("silent", 1),
            "seed": self.params.get("seed", 1),
            "verbosity": 0,
        }

        if "lambda" in self.params:
            self.learner_params["lambda"] = self.params["lambda"]
        if "alpha" in self.params:
            self.learner_params["alpha"] = self.params["alpha"]

        # check https://github.com/dmlc/xgboost/issues/5637
        if self.learner_params["seed"] > 2147483647:
            self.learner_params["seed"] = self.learner_params["seed"] % 2147483647
        if "num_class" in self.params:  # multiclass classification
            self.learner_params["num_class"] = self.params.get("num_class")

        if "max_rounds" in self.params:
            self.boosting_rounds = self.params["max_rounds"]

        self.custom_eval_metric = None
        if self.params.get("eval_metric", "") == "r2":
            self.custom_eval_metric = xgboost_eval_metric_r2
        elif self.params.get("eval_metric", "") == "spearman":
            self.custom_eval_metric = xgboost_eval_metric_spearman
        elif self.params.get("eval_metric", "") == "pearson":
            self.custom_eval_metric = xgboost_eval_metric_pearson
        elif self.params.get("eval_metric", "") == "f1":
            self.custom_eval_metric = xgboost_eval_metric_f1
        elif self.params.get("eval_metric", "") == "average_precision":
            self.custom_eval_metric = xgboost_eval_metric_average_precision
        elif self.params.get("eval_metric", "") == "accuracy":
            self.custom_eval_metric = xgboost_eval_metric_accuracy
        elif self.params.get("eval_metric", "") == "mse":
            self.custom_eval_metric = xgboost_eval_metric_mse
        elif self.params.get("eval_metric", "") == "user_defined_metric":
            self.custom_eval_metric = xgboost_eval_metric_user_defined

        self.best_ntree_limit = 0
        logger.debug("XgbLearner __init__")

    """
    def get_boosting_rounds(self, dtrain, evals, esr, max_time):
        if max_time is None:
            return self.boosting_rounds

        start_time = time.time()
        evals_result = {}
        model = xgb.train(
            self.learner_params,
            dtrain,
            2,
            evals=evals,
            early_stopping_rounds=esr,
            evals_result=evals_result,
            verbose_eval=False,
        )
        time_1_iter = (time.time() - start_time) / 2.0

        # 2.0 is just a scaling factor
        # purely heuristic
        iters = int(max_time / time_1_iter * 2.0)
        iters = max(iters, 100)
        iters = min(iters, 10000)
        return iters
    """

    def fit(
        self,
        X,
        y,
        sample_weight=None,
        X_validation=None,
        y_validation=None,
        sample_weight_validation=None,
        log_to_file=None,
        max_time=None,
    ):
        dtrain = xgb.DMatrix(
            X.values if isinstance(X, pd.DataFrame) else X,
            label=y,
            missing=np.NaN,
            weight=sample_weight,
        )
        dvalidation = xgb.DMatrix(
            X_validation.values
            if isinstance(X_validation, pd.DataFrame)
            else X_validation,
            label=y_validation,
            missing=np.NaN,
            weight=sample_weight_validation,
        )
        evals_result = {}

        evals = []
        esr = None
        if X_validation is not None and y_validation is not None:
            evals = [(dtrain, "train"), (dvalidation, "validation")]
            esr = self.early_stopping_rounds

        # disable for now, dont have better idea how to handle time limit ...
        # looks like there is better not to limit the algorithm
        # just wait till they converge ...
        # boosting_rounds = self.get_boosting_rounds(dtrain, evals, esr, max_time)

        if self.custom_eval_metric is not None:
            del self.learner_params["eval_metric"]

        self.model = xgb.train(
            self.learner_params,
            dtrain,
            self.boosting_rounds,
            evals=evals,
            early_stopping_rounds=esr,
            evals_result=evals_result,
            verbose_eval=False,
            feval=self.custom_eval_metric
            # callbacks=[time_constraint] # callback slows down by factor ~8
        )

        del dtrain
        del dvalidation

        # dump_list = self.model.get_dump()
        # num_trees = len(dump_list)
        # print(self.model.best_ntree_limit, num_trees)

        if log_to_file is not None:

            metric_name = list(evals_result["train"].keys())[-1]

            result = pd.DataFrame(
                {
                    "iteration": range(len(evals_result["train"][metric_name])),
                    "train": evals_result["train"][metric_name],
                    "validation": evals_result["validation"][metric_name],
                }
            )
            # it a is custom metric
            # that is always minimized
            # we need to revert it
            if metric_name in [
                "r2",
                "spearman",
                "pearson",
                "f1",
                "average_precision",
                "accuracy",
            ]:
                result["train"] *= -1.0
                result["validation"] *= -1.0

            result.to_csv(log_to_file, index=False, header=False)

        # save best_ntree_limit because of:
        # https://github.com/dmlc/xgboost/issues/805
        self.best_ntree_limit = self.model.best_ntree_limit
        # fix high memory consumption in xgboost,
        # waiting for release with fix
        # https://github.com/dmlc/xgboost/issues/5474
        """
        # disable, for now all learners are saved to hard disk and then deleted from RAM
        with tempfile.NamedTemporaryFile() as tmp:
            self.model.save_model(tmp.name)
            del self.model
            self.model = xgb.Booster()
            self.model.load_model(tmp.name)
        """

    def is_fitted(self):
        return self.model is not None

    def predict(self, X):
        self.reload()

        if self.model is None:
            raise XgbAlgorithmException("Xgboost model is None")

        dtrain = xgb.DMatrix(
            X.values if isinstance(X, pd.DataFrame) else X, missing=np.NaN
        )
        if "iteration_range" in str(signature(self.model.predict)):
            # the newer version
            a = self.model.predict(dtrain, iteration_range=(0, self.best_ntree_limit))
        else:
            # the older interface
            a = self.model.predict(dtrain, ntree_limit=self.best_ntree_limit)

        return a

    def copy(self):
        return copy.deepcopy(self)

    def save(self, model_file_path):
        self.model.save_model(model_file_path)
        self.model_file_path = model_file_path
        logger.debug("XgbAlgorithm save model to %s" % model_file_path)

    def load(self, model_file_path):
        logger.debug("XgbLearner load model from %s" % model_file_path)
        self.model = xgb.Booster()  # init model
        self.model.load_model(model_file_path)
        self.model_file_path = model_file_path

    def file_extension(self):
        return "xgboost"

    def get_metric_name(self):
        metric = self.params.get("eval_metric")
        if metric is None:
            return None
        if metric == "mlogloss":
            return "logloss"
        return metric


# For binary classification target should be 0, 1. There should be no NaNs in target.
xgb_bin_class_params = {
    "objective": ["binary:logistic"],
    "eta": [0.05, 0.075, 0.1, 0.15],
    "max_depth": [4, 5, 6, 7, 8, 9],
    "min_child_weight": [1, 5, 10, 25, 50],
    "subsample": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
}

classification_bin_default_params = {
    "objective": "binary:logistic",
    "eta": 0.075,
    "max_depth": 6,
    "min_child_weight": 1,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
}

xgb_regression_params = dict(xgb_bin_class_params)
xgb_regression_params["objective"] = ["reg:squarederror"]
# xgb_regression_params["eval_metric"] = ["rmse", "mae", "mape"]
xgb_regression_params["max_depth"] = [4, 5, 6, 7, 8, 9]


xgb_multi_class_params = dict(xgb_bin_class_params)
xgb_multi_class_params["objective"] = ["multi:softprob"]
# xgb_multi_class_params["eval_metric"] = ["mlogloss"]

classification_multi_default_params = {
    "objective": "multi:softprob",
    "eta": 0.075,
    "max_depth": 6,
    "min_child_weight": 1,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
}


regression_default_params = {
    "objective": "reg:squarederror",
    "eta": 0.075,
    "max_depth": 6,
    "min_child_weight": 1,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
}

additional = {
    "max_rounds": 10000,
    "early_stopping_rounds": 50,
    "max_rows_limit": None,
    "max_cols_limit": None,
}
required_preprocessing = [
    "missing_values_inputation",
    "convert_categorical",
    "datetime_transform",
    "text_transform",
    "target_as_integer",
]

AlgorithmsRegistry.add(
    BINARY_CLASSIFICATION,
    XgbAlgorithm,
    xgb_bin_class_params,
    required_preprocessing,
    additional,
    classification_bin_default_params,
)

AlgorithmsRegistry.add(
    MULTICLASS_CLASSIFICATION,
    XgbAlgorithm,
    xgb_multi_class_params,
    required_preprocessing,
    additional,
    classification_multi_default_params,
)

regression_required_preprocessing = [
    "missing_values_inputation",
    "convert_categorical",
    "datetime_transform",
    "text_transform",
    "target_scale",
]


AlgorithmsRegistry.add(
    REGRESSION,
    XgbAlgorithm,
    xgb_regression_params,
    regression_required_preprocessing,
    additional,
    regression_default_params,
)
