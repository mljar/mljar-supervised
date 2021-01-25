import logging
import copy
import numpy as np
import pandas as pd
import os
import time
import xgboost as xgb

from supervised.algorithms.algorithm import BaseAlgorithm
from supervised.algorithms.registry import AlgorithmsRegistry
from supervised.algorithms.registry import (
    BINARY_CLASSIFICATION,
    MULTICLASS_CLASSIFICATION,
    REGRESSION,
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
            # "silent": self.params.get("silent", 1),
            "seed": self.params.get("seed", 1),
        }

        # check https://github.com/dmlc/xgboost/issues/5637
        if self.learner_params["seed"] > 2147483647:
            self.learner_params["seed"] = self.learner_params["seed"] % 2147483647
        if "num_class" in self.params:  # multiclass classification
            self.learner_params["num_class"] = self.params.get("num_class")

        self.best_ntree_limit = 0
        logger.debug("XgbLearner __init__")

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

    def fit(
        self,
        X,
        y,
        sample_weight=None,
        X_validation=None,
        y_validation=None,
        sample_weight_validation=None,
        log_to_file=None,
        max_time=None
    ):
        dtrain = xgb.DMatrix(
            X.to_numpy() if isinstance(X, pd.DataFrame) else X,
            label=y,
            missing=np.NaN,
            weight=sample_weight,
        )
        dvalidation = xgb.DMatrix(
            X_validation.to_numpy()
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

        boosting_rounds = self.get_boosting_rounds(dtrain, evals, esr, max_time)

        self.model = xgb.train(
            self.learner_params,
            dtrain,
            boosting_rounds,
            evals=evals,
            early_stopping_rounds=esr,
            evals_result=evals_result,
            verbose_eval=False
            # callbacks=[time_constraint] # callback slows down by factor ~8
        )

        #dump_list = self.model.get_dump()
        #num_trees = len(dump_list)
        #print(self.model.best_ntree_limit, num_trees)

        if log_to_file is not None:
            metric_name = list(evals_result["train"].keys())[0]
            result = pd.DataFrame(
                {
                    "iteration": range(len(evals_result["train"][metric_name])),
                    "train": evals_result["train"][metric_name],
                    "validation": evals_result["validation"][metric_name],
                }
            )
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

    def predict(self, X):
        self.reload()

        if self.model is None:
            raise XgbAlgorithmException("Xgboost model is None")

        dtrain = xgb.DMatrix(
            X.to_numpy() if isinstance(X, pd.DataFrame) else X, missing=np.NaN
        )
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

    def get_params(self):
        return {
            "library_version": self.library_version,
            "algorithm_name": self.algorithm_name,
            "algorithm_short_name": self.algorithm_short_name,
            "uid": self.uid,
            "params": self.params,
            "best_ntree_limit": self.best_ntree_limit,
        }

    def set_params(self, json_desc):
        self.library_version = json_desc.get("library_version", self.library_version)
        self.algorithm_name = json_desc.get("algorithm_name", self.algorithm_name)
        self.algorithm_short_name = json_desc.get(
            "algorithm_short_name", self.algorithm_short_name
        )
        self.uid = json_desc.get("uid", self.uid)
        self.params = json_desc.get("params", self.params)
        self.best_ntree_limit = json_desc.get("best_ntree_limit", self.best_ntree_limit)

    def file_extension(self):
        return "xgboost"

    def get_metric_name(self):
        metric = self.params.get("eval_metric")
        if metric is None:
            return None
        if metric == "logloss":
            return "logloss"
        elif metric == "auc":
            return "auc"
        elif metric == "mlogloss":
            return "logloss"
        elif metric == "rmse":
            return "rmse"
        return None


# For binary classification target should be 0, 1. There should be no NaNs in target.
xgb_bin_class_params = {
    "objective": ["binary:logistic"],
    "eval_metric": ["logloss", "auc"],
    "eta": [0.05, 0.075, 0.1, 0.15],
    "max_depth": [4, 5, 6, 7, 8, 9],
    "min_child_weight": [1, 5, 10, 25, 50],
    "subsample": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
}

classification_bin_default_params = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "eta": 0.075,
    "max_depth": 6,
    "min_child_weight": 1,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
}

xgb_regression_params = dict(xgb_bin_class_params)
xgb_regression_params["objective"] = ["reg:squarederror"]
xgb_regression_params["eval_metric"] = ["rmse", "mae"]
xgb_regression_params["max_depth"] = [4, 5, 6, 7, 8, 9]


xgb_multi_class_params = dict(xgb_bin_class_params)
xgb_multi_class_params["objective"] = ["multi:softprob"]
xgb_multi_class_params["eval_metric"] = ["mlogloss"]

classification_multi_default_params = {
    "objective": "multi:softprob",
    "eval_metric": "mlogloss",
    "eta": 0.075,
    "max_depth": 6,
    "min_child_weight": 1,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
}


regression_default_params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
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
