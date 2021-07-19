import logging
import copy
import numpy as np
import pandas as pd
import time
import os
import contextlib
import lightgbm as lgb

from supervised.algorithms.algorithm import BaseAlgorithm
from supervised.algorithms.registry import AlgorithmsRegistry
from supervised.algorithms.registry import (
    BINARY_CLASSIFICATION,
    MULTICLASS_CLASSIFICATION,
    REGRESSION,
)
from supervised.utils.metric import (
    lightgbm_eval_metric_r2,
    lightgbm_eval_metric_spearman,
    lightgbm_eval_metric_pearson,
    lightgbm_eval_metric_f1,
    lightgbm_eval_metric_average_precision,
    lightgbm_eval_metric_accuracy,
    lightgbm_eval_metric_user_defined,
)
from supervised.utils.config import LOG_LEVEL

logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)


def lightgbm_objective(ml_task, automl_eval_metric):
    objective = "regression"
    if ml_task == BINARY_CLASSIFICATION:
        objective = "binary"
    elif ml_task == MULTICLASS_CLASSIFICATION:
        objective = "multiclass"
    else:  # ml_task == REGRESSION
        objective = "regression"
    return objective


def lightgbm_eval_metric(ml_task, automl_eval_metric):
    if automl_eval_metric == "user_defined_metric":
        return "custom", automl_eval_metric
    metric_name_mapping = {
        BINARY_CLASSIFICATION: {
            "auc": "auc",
            "logloss": "binary_logloss",
            "f1": "custom",
            "average_precision": "custom",
            "accuracy": "custom",
        },
        MULTICLASS_CLASSIFICATION: {
            "logloss": "multi_logloss",
            "f1": "custom",
            "accuracy": "custom",
        },
        REGRESSION: {
            "rmse": "rmse",
            "mse": "l2",
            "mae": "l1",
            "mape": "mape",
            "r2": "custom",
            "spearman": "custom",
            "pearson": "custom",
        },
    }

    metric = metric_name_mapping[ml_task][automl_eval_metric]
    custom_eval_metric = None

    if automl_eval_metric in [
        "r2",
        "spearman",
        "pearson",
        "f1",
        "average_precision",
        "accuracy",
    ]:
        custom_eval_metric = automl_eval_metric

    return metric, custom_eval_metric


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

        n_jobs = self.params.get("n_jobs", 0)
        # 0 is the default for LightGBM to use all cores
        if n_jobs == -1:
            n_jobs = 0

        self.learner_params = {
            "boosting_type": "gbdt",
            "objective": self.params.get("objective", "binary"),
            "metric": self.params.get("metric", "binary_logloss"),
            "num_leaves": self.params.get("num_leaves", 31),
            "learning_rate": self.params.get("learning_rate", 0.1),
            "feature_fraction": self.params.get("feature_fraction", 1.0),
            "bagging_fraction": self.params.get("bagging_fraction", 1.0),
            "min_data_in_leaf": self.params.get("min_data_in_leaf", 20),
            "num_threads": n_jobs,
            "verbose": -1,
            "seed": self.params.get("seed", 1),
            "extra_trees": self.params.get("extra_trees", False),
        }

        for extra_param in [
            "lambda_l1",
            "lambda_l2",
            "bagging_freq",
            "feature_pre_filter",
            "cat_feature",
            "cat_l2",
            "cat_smooth",
            "max_bin",
        ]:
            if extra_param in self.params:
                self.learner_params[extra_param] = self.params[extra_param]

        if "num_boost_round" in self.params:
            self.rounds = self.params["num_boost_round"]
        if "early_stopping_rounds" in self.params:
            self.early_stopping_rounds = self.params["early_stopping_rounds"]

        if "num_class" in self.params:  # multiclass classification
            self.learner_params["num_class"] = self.params.get("num_class")

        self.custom_eval_metric = None
        if self.params.get("custom_eval_metric_name") is not None:
            if self.params["custom_eval_metric_name"] == "r2":
                self.custom_eval_metric = lightgbm_eval_metric_r2
            elif self.params["custom_eval_metric_name"] == "spearman":
                self.custom_eval_metric = lightgbm_eval_metric_spearman
            elif self.params["custom_eval_metric_name"] == "pearson":
                self.custom_eval_metric = lightgbm_eval_metric_pearson
            elif self.params["custom_eval_metric_name"] == "f1":
                self.custom_eval_metric = lightgbm_eval_metric_f1
            elif self.params["custom_eval_metric_name"] == "average_precision":
                self.custom_eval_metric = lightgbm_eval_metric_average_precision
            elif self.params["custom_eval_metric_name"] == "accuracy":
                self.custom_eval_metric = lightgbm_eval_metric_accuracy
            elif self.params["custom_eval_metric_name"] == "user_defined_metric":
                self.custom_eval_metric = lightgbm_eval_metric_user_defined

        logger.debug("LightgbmLearner __init__")

    def file_extension(self):
        return "lightgbm"

    def update(self, update_params):
        pass

    """
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
        lgb_train = lgb.Dataset(
            X.values if isinstance(X, pd.DataFrame) else X,
            y,
            weight=sample_weight,
        )
        valid_sets = None
        if self.early_stopping_rounds == 0:
            self.model = lgb.train(
                self.learner_params,
                lgb_train,
                num_boost_round=self.rounds,
                init_model=self.model,
            )
        else:

            valid_names = None
            esr = None
            if X_validation is not None and y_validation is not None:
                valid_sets = [
                    lgb_train,
                    lgb.Dataset(
                        X_validation.values
                        if isinstance(X_validation, pd.DataFrame)
                        else X_validation,
                        y_validation,
                        weight=sample_weight_validation,
                    ),
                ]
                valid_names = ["train", "validation"]
                esr = self.early_stopping_rounds
            evals_result = {}

            # disable for now ...
            # boosting_rounds = self.get_boosting_rounds(lgb_train, valid_sets, esr, max_time)

            self.model = lgb.train(
                self.learner_params,
                lgb_train,
                num_boost_round=self.rounds,
                valid_sets=valid_sets,
                valid_names=valid_names,
                early_stopping_rounds=esr,
                evals_result=evals_result,
                verbose_eval=False,
                feval=self.custom_eval_metric,
            )

            del lgb_train
            if valid_sets is not None:
                del valid_sets[0]
                del valid_sets

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

    def is_fitted(self):
        return self.model is not None

    def predict(self, X):
        self.reload()
        return self.model.predict(X.values if isinstance(X, pd.DataFrame) else X)

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

    def get_metric_name(self):
        metric = self.params.get("metric")
        custom_metric = self.params.get("custom_eval_metric_name")

        if metric is None:
            return None
        if metric == "custom":
            return custom_metric
        if metric == "binary_logloss":
            return "logloss"
        elif metric == "multi_logloss":
            return "logloss"
        return metric


lgbm_bin_params = {
    "objective": ["binary"],
    "num_leaves": [15, 31, 63, 95, 127],
    "learning_rate": [0.05, 0.1, 0.2],
    "feature_fraction": [0.5, 0.8, 0.9, 1.0],
    "bagging_fraction": [0.5, 0.8, 0.9, 1.0],
    "min_data_in_leaf": [5, 10, 15, 20, 30, 50],
}

classification_bin_default_params = {
    "objective": "binary",
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

classification_multi_default_params = {
    "objective": "multiclass",
    "num_leaves": 63,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "min_data_in_leaf": 10,
}

lgbr_params = copy.deepcopy(lgbm_bin_params)
lgbr_params["objective"] = ["regression"]

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
