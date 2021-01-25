import logging
import copy
import numpy as np
import pandas as pd
import time
import os
import contextlib
import multiprocessing
import lightgbm as lgb

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


class LightgbmAlgorithm(BaseAlgorithm):

    algorithm_name = "LightGBM"
    algorithm_short_name = "LightGBM"

    def __init__(self, params):
        super(LightgbmAlgorithm, self).__init__(params)
        self.library_version = lgb.__version__

        self.explain_level = params.get("explain_level", 0)
        self.rounds = additional.get("max_rounds", 10000)
        self.max_iters = 1
        self.early_stopping_rounds = additional.get("early_stopping_rounds", 50)

        self.learner_params = {
            "boosting_type": "gbdt",
            "objective": self.params.get("objective", "binary"),
            "metric": self.params.get("metric", "binary_logloss"),
            "num_threads": multiprocessing.cpu_count(),
            "num_leaves": self.params.get("num_leaves", 31),
            "learning_rate": self.params.get("learning_rate", 0.1),
            "feature_fraction": self.params.get("feature_fraction", 1.0),
            "bagging_fraction": self.params.get("bagging_fraction", 1.0),
            "min_data_in_leaf": self.params.get("min_data_in_leaf", 20),
            "verbose": -1,
            "seed": self.params.get("seed", 1),
        }
        if "num_class" in self.params:  # multiclass classification
            self.learner_params["num_class"] = self.params.get("num_class")

        logger.debug("LightgbmLearner __init__")

    def file_extension(self):
        return "lightgbm"

    def update(self, update_params):
        pass

    def get_boosting_rounds(self, lgb_train, valid_sets, esr, max_time):
        if max_time is None:
            max_time = 3600.0
        start_time = time.time()
        evals_result = {}
        model = lgb.train(
            self.learner_params,
            lgb_train,
            num_boost_round=2,
            valid_sets=valid_sets,
            early_stopping_rounds=esr,
            evals_result=evals_result,
            verbose_eval=False
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
        lgb_train = lgb.Dataset(
            X.to_numpy() if isinstance(X, pd.DataFrame) else X, y, weight=sample_weight
        )
        if self.early_stopping_rounds == 0:
            self.model = lgb.train(
                self.learner_params,
                lgb_train,
                num_boost_round=self.rounds,
                init_model=self.model,
            )
        else:
            valid_sets = None
            valid_names = None
            esr = None
            if X_validation is not None and y_validation is not None:
                valid_sets = [
                    lgb_train,
                    lgb.Dataset(
                        X_validation.to_numpy()
                        if isinstance(X_validation, pd.DataFrame)
                        else X_validation,
                        y_validation,
                        weight=sample_weight_validation,
                    ),
                ]
                valid_names = ["train", "validation"]
                esr = self.early_stopping_rounds
            evals_result = {}

            boosting_rounds = self.get_boosting_rounds(lgb_train, valid_sets, esr, max_time)            

            self.model = lgb.train(
                self.learner_params,
                lgb_train,
                num_boost_round=boosting_rounds,
                #init_model=self.model,
                valid_sets=valid_sets,
                valid_names=valid_names,
                early_stopping_rounds=esr,
                evals_result=evals_result,
                verbose_eval=False,
            )

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

    def predict(self, X):
        self.reload()
        return self.model.predict(X.to_numpy() if isinstance(X, pd.DataFrame) else X)

    def copy(self):
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            return copy.deepcopy(self)

    def save(self, model_file_path):
        self.model.save_model(model_file_path)
        self.model_file_path = model_file_path
        logger.debug("LightgbmAlgorithm save model to %s" % model_file_path)

    def load(self, model_file_path):
        logger.debug("LightgbmAlgorithm load model from %s" % model_file_path)
        self.model_file_path = model_file_path
        self.model = lgb.Booster(model_file=model_file_path)

    def get_params(self):
        json_desc = {
            "library_version": self.library_version,
            "algorithm_name": self.algorithm_name,
            "algorithm_short_name": self.algorithm_short_name,
            "uid": self.uid,
            "params": self.params,
        }
        return json_desc

    def set_params(self, json_desc):
        self.library_version = json_desc.get("library_version", self.library_version)
        self.algorithm_name = json_desc.get("algorithm_name", self.algorithm_name)
        self.algorithm_short_name = json_desc.get(
            "algorithm_short_name", self.algorithm_short_name
        )
        self.uid = json_desc.get("uid", self.uid)
        self.params = json_desc.get("params", self.params)

    def get_metric_name(self):
        metric = self.params.get("metric")
        if metric is None:
            return None
        if metric == "binary_logloss":
            return "logloss"
        elif metric == "auc":
            return "auc"
        elif metric == "multi_logloss":
            return "logloss"
        elif metric == "rmse":
            return "rmse"
        return None


lgbm_bin_params = {
    "objective": ["binary"],
    "metric": ["binary_logloss", "auc"],
    "num_leaves": [15, 31, 63, 95, 127],
    "learning_rate": [0.05, 0.1, 0.2],
    "feature_fraction": [0.5, 0.8, 0.9, 1.0],
    "bagging_fraction": [0.5, 0.8, 0.9, 1.0],
    "min_data_in_leaf": [5, 10, 15, 20, 30, 50],
}

classification_bin_default_params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "num_leaves": 63,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "min_data_in_leaf": 10,
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

lgbm_multi_params = copy.deepcopy(lgbm_bin_params)
lgbm_multi_params["objective"] = ["multiclass"]
lgbm_multi_params["metric"] = ["multi_logloss"]

classification_multi_default_params = {
    "objective": "multiclass",
    "metric": "multi_logloss",
    "num_leaves": 63,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "min_data_in_leaf": 10,
}

lgbr_params = copy.deepcopy(lgbm_bin_params)
lgbr_params["objective"] = ["regression"]
lgbr_params["metric"] = ["rmse"]

AlgorithmsRegistry.add(
    BINARY_CLASSIFICATION,
    LightgbmAlgorithm,
    lgbm_bin_params,
    required_preprocessing,
    additional,
    classification_bin_default_params,
)

AlgorithmsRegistry.add(
    MULTICLASS_CLASSIFICATION,
    LightgbmAlgorithm,
    lgbm_multi_params,
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


regression_default_params = {
    "objective": "regression",
    "metric": "rmse",
    "num_leaves": 63,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "min_data_in_leaf": 10,
}

AlgorithmsRegistry.add(
    REGRESSION,
    LightgbmAlgorithm,
    lgbr_params,
    regression_required_preprocessing,
    additional,
    regression_default_params,
)
