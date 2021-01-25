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
from supervised.utils.config import LOG_LEVEL

logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)

from catboost import CatBoostClassifier, CatBoostRegressor, CatBoost, Pool
import catboost


class CatBoostAlgorithm(BaseAlgorithm):

    algorithm_name = "CatBoost"
    algorithm_short_name = "CatBoost"

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

        self.learner_params = {
            "learning_rate": self.params.get("learning_rate", 0.1),
            "depth": self.params.get("depth", 3),
            "rsm": self.params.get("rsm", 1.0),
            "random_seed": self.params.get("seed", 1),
            "loss_function": loss_function,
        }

        self.model = Algo(
            iterations=self.rounds,
            learning_rate=self.learner_params["learning_rate"],
            depth=self.learner_params["depth"],
            rsm=self.learner_params["rsm"],
            loss_function=self.learner_params["loss_function"],
            verbose=False,
            allow_writing_files=False,
        )
        self.cat_features = None

        logger.debug("CatBoostAlgorithm.__init__")

    def _assess_iterations(self, X, y, eval_set, max_time):
        if max_time is None:
            max_time = 3600
        try:
            model = copy.deepcopy(self.model)
            model.set_params(iterations=1)
            start_time = time.time()
            model.fit(
                X,
                y,
                cat_features=self.cat_features,
                init_model=None if self.model.tree_count_ is None else self.model,
                eval_set=eval_set,
                early_stopping_rounds=self.early_stopping_rounds,
                verbose_eval=False,
            )
            elapsed_time = np.round(time.time() - start_time, 2)
            new_rounds = int(min(10000, max_time / elapsed_time * 2.0))
            new_rounds = max(max_rounds, 100)
            return new_rounds
        except Exception as e:
            return 1000

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
        if self.model.tree_count_ is not None:
            print("CatBoost model already fitted. Skip fit().")
            return

        if self.cat_features is None:
            self.cat_features = []
            for i in range(X.shape[1]):
                if PreprocessingUtils.is_categorical(X.iloc[:, i]):
                    self.cat_features += [i]

        eval_set = None
        if X_validation is not None and y_validation is not None:
            eval_set = Pool(
                data=X_validation,
                label=y_validation,
                cat_features=self.cat_features,
                weight=sample_weight_validation,
            )

        new_iterations = self._assess_iterations(X, y, eval_set, max_time)
        self.model.set_params(iterations=new_iterations)

        self.model.fit(
            X,
            y,
            sample_weight=sample_weight,
            cat_features=self.cat_features,
            init_model=None if self.model.tree_count_ is None else self.model,
            eval_set=eval_set,
            early_stopping_rounds=self.early_stopping_rounds,
            verbose_eval=False,
        )
        if log_to_file is not None:

            metric_name = list(self.model.evals_result_["learn"].keys())[0]
            result = pd.DataFrame(
                {
                    "iteration": range(
                        len(self.model.evals_result_["learn"][metric_name])
                    ),
                    "train": self.model.evals_result_["learn"][metric_name],
                    "validation": self.model.evals_result_["validation"][metric_name],
                }
            )
            result.to_csv(log_to_file, index=False, header=False)

    def predict(self, X):
        self.reload()
        if self.params["ml_task"] == BINARY_CLASSIFICATION:
            return self.model.predict_proba(X)[:, 1]
        elif self.params["ml_task"] == MULTICLASS_CLASSIFICATION:
            return self.model.predict_proba(X)
        return self.model.predict(X)

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

        self.model = Algo().load_model(model_file_path)
        self.model_file_path = model_file_path

    def get_params(self):
        return {
            "library_version": self.library_version,
            "algorithm_name": self.algorithm_name,
            "algorithm_short_name": self.algorithm_short_name,
            "uid": self.uid,
            "params": self.params,
        }

    def set_params(self, json_desc):
        self.library_version = json_desc.get("library_version", self.library_version)
        self.algorithm_name = json_desc.get("algorithm_name", self.algorithm_name)
        self.algorithm_short_name = json_desc.get(
            "algorithm_short_name", self.algorithm_short_name
        )
        self.uid = json_desc.get("uid", self.uid)
        self.params = json_desc.get("params", self.params)

    def file_extension(self):
        return "catboost"

    def get_metric_name(self):
        metric = self.params.get("loss_function")
        if metric is None:
            return None
        if metric == "Logloss":
            return "logloss"
        elif metric == "MultiClass":
            return "logloss"
        elif metric == "RMSE":
            return "rmse"
        return None


classification_params = {
    "learning_rate": [0.05, 0.1, 0.2],
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
multiclass_classification_params["learning_rate"] = [0.1, 0.2]

multiclass_classification_default_params = copy.deepcopy(classification_default_params)
multiclass_classification_default_params["loss_function"] = "MultiClass"
multiclass_classification_default_params["depth"] = 5
multiclass_classification_default_params["learning_rate"] = 0.1


AlgorithmsRegistry.add(
    MULTICLASS_CLASSIFICATION,
    CatBoostAlgorithm,
    multiclass_classification_params,
    required_preprocessing,
    additional,
    multiclass_classification_default_params,
)

regression_params = copy.deepcopy(classification_params)
regression_params["loss_function"] = ["RMSE", "MAE"]

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
