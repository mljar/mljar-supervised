import logging
import os
import sklearn
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

from supervised.algorithms.algorithm import BaseAlgorithm
from supervised.algorithms.sklearn import (
    SklearnTreesEnsembleClassifierAlgorithm,
    SklearnTreesEnsembleRegressorAlgorithm,
)
from supervised.algorithms.registry import AlgorithmsRegistry
from supervised.algorithms.registry import (
    BINARY_CLASSIFICATION,
    MULTICLASS_CLASSIFICATION,
    REGRESSION,
)
from supervised.utils.config import LOG_LEVEL

logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)


class ExtraTreesAlgorithm(SklearnTreesEnsembleClassifierAlgorithm):

    algorithm_name = "Extra Trees Classifier"
    algorithm_short_name = "Extra Trees"

    def __init__(self, params):
        super(ExtraTreesAlgorithm, self).__init__(params)
        logger.debug("ExtraTreesAlgorithm.__init__")

        self.library_version = sklearn.__version__
        self.trees_in_step = additional.get("trees_in_step", 100)
        self.max_steps = additional.get("max_steps", 50)
        self.early_stopping_rounds = additional.get("early_stopping_rounds", 50)
        self.model = ExtraTreesClassifier(
            n_estimators=self.trees_in_step,
            criterion=params.get("criterion", "gini"),
            max_features=params.get("max_features", 0.8),
            max_depth=params.get("max_depth", 6),
            min_samples_split=params.get("min_samples_split", 4),
            warm_start=True,
            n_jobs=-1,
            random_state=params.get("seed", 1),
        )

    def file_extension(self):
        return "extra_trees"


class ExtraTreesRegressorAlgorithm(SklearnTreesEnsembleRegressorAlgorithm):

    algorithm_name = "Extra Trees Regressor"
    algorithm_short_name = "Extra Trees"

    def __init__(self, params):
        super(ExtraTreesRegressorAlgorithm, self).__init__(params)
        logger.debug("ExtraTreesRegressorAlgorithm.__init__")

        self.library_version = sklearn.__version__
        self.trees_in_step = regression_additional.get("trees_in_step", 100)
        self.max_steps = regression_additional.get("max_steps", 50)
        self.early_stopping_rounds = regression_additional.get(
            "early_stopping_rounds", 50
        )
        self.model = ExtraTreesRegressor(
            n_estimators=self.trees_in_step,
            criterion=params.get("criterion", "mse"),
            max_features=params.get("max_features", 0.6),
            max_depth=params.get("max_depth", 6),
            min_samples_split=params.get("min_samples_split", 30),
            warm_start=True,
            n_jobs=-1,
            random_state=params.get("seed", 1),
        )

    def file_extension(self):
        return "extra_trees"


# For binary classification target should be 0, 1. There should be no NaNs in target.
et_params = {
    "criterion": ["gini", "entropy"],
    "max_features": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "min_samples_split": [10, 20, 30, 40, 50],
    "max_depth": [3, 4, 5, 6, 7],
}

classification_default_params = {
    "criterion": "gini",
    "max_features": 0.9,
    "min_samples_split": 30,
    "max_depth": 4,
}

additional = {
    "trees_in_step": 100,
    "max_steps": 50,
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
    ExtraTreesAlgorithm,
    et_params,
    required_preprocessing,
    additional,
    classification_default_params,
)

AlgorithmsRegistry.add(
    MULTICLASS_CLASSIFICATION,
    ExtraTreesAlgorithm,
    et_params,
    required_preprocessing,
    additional,
    classification_default_params,
)


#
# REGRESSION
#

regression_et_params = {
    "criterion": [
        "mse"
    ],  # remove "mae" because it slows down a lot https://github.com/scikit-learn/scikit-learn/issues/9626
    "max_features": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "min_samples_split": [10, 20, 30, 40, 50],
    "max_depth": [3, 4, 5, 6, 7],
}

regression_default_params = {
    "criterion": "mse",
    "max_features": 0.9,
    "min_samples_split": 30,
    "max_depth": 4,
}

regression_additional = {
    "trees_in_step": 100,
    "max_steps": 50,
    "early_stopping_rounds": 50,
    "max_rows_limit": None,
    "max_cols_limit": None,
}
regression_required_preprocessing = [
    "missing_values_inputation",
    "convert_categorical",
    "datetime_transform",
    "text_transform",
    "target_scale",
]

AlgorithmsRegistry.add(
    REGRESSION,
    ExtraTreesRegressorAlgorithm,
    regression_et_params,
    regression_required_preprocessing,
    regression_additional,
    regression_default_params,
)
