import logging
import copy
import numpy as np
import pandas as pd
import os
import time

from supervised.algorithms.algorithm import BaseAlgorithm
from supervised.algorithms.registry import AlgorithmsRegistry
from supervised.algorithms.registry import (
    BINARY_CLASSIFICATION,
    MULTICLASS_CLASSIFICATION,
    REGRESSION,
)
from supervised.preprocessing.preprocessing_utils import PreprocessingUtils
from supervised.utils.metric import (
    CatBoostEvalMetricSpearman,
    CatBoostEvalMetricPearson,
    CatBoostEvalMetricAveragePrecision,
    CatBoostEvalMetricMSE,
    CatBoostEvalMetricUserDefined,
)

from supervised.utils.config import LOG_LEVEL

logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)

from catboost import CatBoostClassifier, CatBoostRegressor, CatBoost, Pool
import catboost


def catboost_eval_metric(ml_task, eval_metric):
    if eval_metric == "user_defined_metric":
        return eval_metric
    metric_name_mapping = {
        BINARY_CLASSIFICATION: {
            "auc": "AUC",
            "logloss": "Logloss",
            "f1": "F1",
            "average_precision": "average_precision",
            "accuracy": "Accuracy",
        },
        MULTICLASS_CLASSIFICATION: {
            "logloss": "MultiClass",
            "f1": "TotalF1:average=Micro",
            "accuracy": "Accuracy",
        },
        REGRESSION: {
            "rmse": "RMSE",
            "mse": "mse",
            "mae": "MAE",
            "mape": "MAPE",
            "r2": "R2",
            "spearman": "spearman",
            "pearson": "pearson",
        },
    }
    return metric_name_mapping[ml_task][eval_metric]


def catboost_objective(ml_task, eval_metric):
    objective = "RMSE"
    if ml_task == BINARY_CLASSIFICATION:
        objective = "Logloss"
    elif ml_task == MULTICLASS_CLASSIFICATION:
        objective = "MultiClass"
    else:  # ml_task == REGRESSION
        objective = catboost_eval_metric(REGRESSION, eval_metric)
        if objective in [
            "mse",
            "R2",
            "spearman",
            "pearson",
            "user_defined_metric",
        ]:  # cant optimize them directly
            objective = "RMSE"
    return objective


class CatBoostAlgorithm(BaseAlgorithm):

    algorithm_name = "CatBoost"
    algorithm_short_name = "CatBoost"
    warmup_iterations = 20

    def __init__(self, params):
        super(CatBoostAlgorithm, self).__init__(params)
        self.library_version = catboost.__version__
        self.snapshot_file_path = "training_snapshot"

        self.explain_level = params.get("explain_level", 0)
        self.rounds = additional.get("max_rounds", 10000)
        self.max_iters = 1
        self.early_stopping_rounds = additional.get("early_stopping_rounds", 50)

        Algo = CatBoostClassifier
        loss_function = "Logloss"
        if self.params["ml_task"] == BINARY_CLASSIFICATION:
            loss_function = self.params.get("loss_function", "Logloss")
        elif self.params["ml_task"] == MULTICLASS_CLASSIFICATION:
            loss_function = self.params.get("loss_function", "MultiClass")
        elif self.params["ml_task"] == REGRESSION:
            loss_function = self.params.get("loss_function", "RMSE")
            Algo = CatBoostRegressor

        cat_params = {
            "iterations": self.params.get("num_boost_round", self.rounds),
            "learning_rate": self.params.get("learning_rate", 0.1),
            "depth": self.params.get("depth", 3),
            "rsm": self.params.get("rsm", 1.0),
            "l2_leaf_reg": self.params.get("l2_leaf_reg", 3.0),
            "random_strength": self.params.get("random_strength", 1.0),
            "loss_function": loss_function,
            "eval_metric": self.params.get("eval_metric", loss_function),
            # "custom_metric": self.params.get("eval_metric", loss_function),
            "thread_count": self.params.get("n_jobs", -1),
            "verbose": False,
            "allow_writing_files": False,
            "random_seed": self.params.get("seed", 1),
        }

        for extra_param in [
            "min_data_in_leaf",
            "bootstrap_type",
            "bagging_temperature",
            "subsample",
            "border_count",
        ]:
            if extra_param in self.params:
                cat_params[extra_param] = self.params[extra_param]

        self.log_metric_name = cat_params["eval_metric"]
        if cat_params["eval_metric"] == "spearman":
            cat_params["eval_metric"] = CatBoostEvalMetricSpearman()
            self.log_metric_name = "CatBoostEvalMetricSpearman"
        elif cat_params["eval_metric"] == "pearson":
            cat_params["eval_metric"] = CatBoostEvalMetricPearson()
            self.log_metric_name = "CatBoostEvalMetricPearson"
        elif cat_params["eval_metric"] == "average_precision":
            cat_params["eval_metric"] = CatBoostEvalMetricAveragePrecision()
            self.log_metric_name = "CatBoostEvalMetricAveragePrecision"
        elif cat_params["eval_metric"] == "mse":
            cat_params["eval_metric"] = CatBoostEvalMetricMSE()
            self.log_metric_name = "CatBoostEvalMetricMSE"
        elif cat_params["eval_metric"] == "user_defined_metric":
            cat_params["eval_metric"] = CatBoostEvalMetricUserDefined()
            self.log_metric_name = "CatBoostEvalMetricUserDefined"

        self.model = Algo(**cat_params)
        self.cat_features = None
        self.best_ntree_limit = 0

        logger.debug("CatBoostAlgorithm.__init__")

    def _assess_iterations(self, X, y, sample_weight, eval_set, max_time=None):
        if max_time is None:
            max_time = 3600
        try:
            model = copy.deepcopy(self.model)
            model.set_params(iterations=self.warmup_iterations)
            start_time = time.time()
            model.fit(
                X,
                y,
                sample_weight=sample_weight,
                cat_features=self.cat_features,
                init_model=None if self.model.tree_count_ is None else self.model,
                eval_set=eval_set,
                early_stopping_rounds=self.early_stopping_rounds,
                verbose_eval=False,
            )
            elapsed_time = (time.time() - start_time) / float(self.warmup_iterations)
            # print(max_time, elapsed_time, max_time / elapsed_time, np.round(time.time() - start_time, 2))
            new_rounds = int(min(10000, max_time / elapsed_time))
            new_rounds = max(new_rounds, 10)
            return model, new_rounds
        except Exception as e:
            # print(str(e))
            return None, 1000

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
        if self.is_fitted():
            print("CatBoost model already fitted. Skip fit().")
            return

        if self.cat_features is None:
            self.cat_features = []
            for i in range(X.shape[1]):
                if PreprocessingUtils.is_categorical(X.iloc[:, i]):
                    self.cat_features += [i]
                    X.iloc[:, i] = X.iloc[:, i].astype(str)
                    if X_validation is not None:
                        X_validation.iloc[:, i] = X_validation.iloc[:, i].astype(str)

        eval_set = None
        if X_validation is not None and y_validation is not None:
            eval_set = Pool(
                data=X_validation,
                label=y_validation,
                cat_features=self.cat_features,
                weight=sample_weight_validation,
            )

        if self.params.get("num_boost_round") is None:
            model_init, new_iterations = self._assess_iterations(
                X, y, sample_weight, eval_set, max_time
            )
            self.model.set_params(iterations=new_iterations)
        else:
            model_init = None
            self.model.set_params(iterations=self.params.get("num_boost_round"))
            self.early_stopping_rounds = self.params.get("early_stopping_rounds", 50)

        self.model.fit(
            X,
            y,
            sample_weight=sample_weight,
            cat_features=self.cat_features,
            init_model=model_init,
            eval_set=eval_set,
            early_stopping_rounds=self.early_stopping_rounds,
            verbose_eval=False,
        )

        if self.model.best_iteration_ is not None:
            if model_init is not None:
                self.best_ntree_limit = (
                    self.model.best_iteration_ + model_init.tree_count_ + 1
                )
            else:
                self.best_ntree_limit = self.model.best_iteration_ + 1

        else:
            # just take all the trees
            # the warm-up trees are already included
            # dont need to add +1
            self.best_ntree_limit = self.model.tree_count_

        if log_to_file is not None:
            train_scores = self.model.evals_result_["learn"].get(self.log_metric_name)
            validation_scores = self.model.evals_result_["validation"].get(
                self.log_metric_name
            )
            if model_init is not None:
                if train_scores is not None:
                    train_scores = (
                        model_init.evals_result_["learn"].get(self.log_metric_name)
                        + train_scores
                    )
                if validation_scores is not None:
                    validation_scores = (
                        model_init.evals_result_["validation"].get(self.log_metric_name)
                        + validation_scores
                    )
            iteration = None
            if train_scores is not None:
                iteration = range(len(validation_scores))
            elif validation_scores is not None:
                iteration = range(len(validation_scores))

            result = pd.DataFrame(
                {
                    "iteration": iteration,
                    "train": train_scores,
                    "validation": validation_scores,
                }
            )
            result.to_csv(log_to_file, index=False, header=False)

    def is_fitted(self):
        return self.model is not None and self.model.tree_count_ is not None

    def predict(self, X):
        self.reload()
        if self.params["ml_task"] == BINARY_CLASSIFICATION:
            return self.model.predict_proba(X, ntree_end=self.best_ntree_limit)[:, 1]
        elif self.params["ml_task"] == MULTICLASS_CLASSIFICATION:
            return self.model.predict_proba(X, ntree_end=self.best_ntree_limit)

        return self.model.predict(X, ntree_end=self.best_ntree_limit)

    def copy(self):
        return copy.deepcopy(self)

    def save(self, model_file_path):
        self.model.save_model(model_file_path)
        self.model_file_path = model_file_path
        logger.debug("CatBoostAlgorithm save model to %s" % model_file_path)

    def load(self, model_file_path):
        logger.debug("CatBoostLearner load model from %s" % model_file_path)

        # waiting for fix https://github.com/catboost/catboost/issues/696
        Algo = CatBoostClassifier
        if self.params["ml_task"] == REGRESSION:
            Algo = CatBoostRegressor

        # loading might throw warnings in the case of custom eval_metric
        # check https://github.com/catboost/catboost/issues/1169
        self.model = Algo().load_model(model_file_path)
        self.model_file_path = model_file_path

    def file_extension(self):
        return "catboost"

    def get_metric_name(self):
        metric = self.params.get("eval_metric")
        if metric is None:
            return None
        if metric == "Logloss":
            return "logloss"
        elif metric == "AUC":
            return "auc"
        elif metric == "MultiClass":
            return "logloss"
        elif metric == "RMSE":
            return "rmse"
        elif metric == "MSE":
            return "mse"
        elif metric == "MAE":
            return "mae"
        elif metric == "MAPE":
            return "mape"
        elif metric in ["F1", "TotalF1:average=Micro"]:
            return "f1"
        elif metric == "Accuracy":
            return "accuracy"
        return metric


classification_params = {
    "learning_rate": [0.025, 0.05, 0.1, 0.2],
    "depth": [4, 5, 6, 7, 8, 9],
    "rsm": [0.7, 0.8, 0.9, 1],  # random subspace method
    "loss_function": ["Logloss"],
}

classification_default_params = {
    "learning_rate": 0.1,
    "depth": 6,
    "rsm": 1,
    "loss_function": "Logloss",
}

additional = {
    "max_rounds": 10000,
    "early_stopping_rounds": 50,
    "max_rows_limit": None,
    "max_cols_limit": None,
}
required_preprocessing = [
    "missing_values_inputation",
    "datetime_transform",
    "text_transform",
    "target_as_integer",
]


AlgorithmsRegistry.add(
    BINARY_CLASSIFICATION,
    CatBoostAlgorithm,
    classification_params,
    required_preprocessing,
    additional,
    classification_default_params,
)

multiclass_classification_params = copy.deepcopy(classification_params)
multiclass_classification_params["loss_function"] = ["MultiClass"]
multiclass_classification_params["depth"] = [3, 4, 5, 6]
multiclass_classification_params["learning_rate"] = [0.1, 0.15, 0.2]

multiclass_classification_default_params = copy.deepcopy(classification_default_params)
multiclass_classification_default_params["loss_function"] = "MultiClass"
multiclass_classification_default_params["depth"] = 5
multiclass_classification_default_params["learning_rate"] = 0.15


AlgorithmsRegistry.add(
    MULTICLASS_CLASSIFICATION,
    CatBoostAlgorithm,
    multiclass_classification_params,
    required_preprocessing,
    additional,
    multiclass_classification_default_params,
)

regression_params = copy.deepcopy(classification_params)
regression_params["loss_function"] = ["RMSE", "MAE", "MAPE"]

regression_required_preprocessing = [
    "missing_values_inputation",
    "datetime_transform",
    "text_transform",
    "target_scale",
]


regression_default_params = {
    "learning_rate": 0.1,
    "depth": 6,
    "rsm": 1,
    "loss_function": "RMSE",
}

AlgorithmsRegistry.add(
    REGRESSION,
    CatBoostAlgorithm,
    regression_params,
    regression_required_preprocessing,
    additional,
    regression_default_params,
)
