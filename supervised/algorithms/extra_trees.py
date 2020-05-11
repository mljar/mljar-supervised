import logging
import os
import sklearn
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

from supervised.algorithms.algorithm import BaseAlgorithm
from supervised.algorithms.sklearn import SklearnTreesClassifierAlgorithm
from supervised.algorithms.sklearn import SklearnTreesRegressorAlgorithm
from supervised.algorithms.registry import AlgorithmsRegistry
from supervised.algorithms.registry import (
    BINARY_CLASSIFICATION,
    MULTICLASS_CLASSIFICATION,
    REGRESSION,
)
from supervised.utils.config import LOG_LEVEL

logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)


class ExtraTreesAlgorithm(SklearnTreesClassifierAlgorithm):

    algorithm_name = "Extra Trees Classifier"
    algorithm_short_name = "Extra Trees"

    def __init__(self, params):
        super(ExtraTreesAlgorithm, self).__init__(params)
        logger.debug("ExtraTreesAlgorithm.__init__")

        self.library_version = sklearn.__version__
        self.trees_in_step = additional.get("trees_in_step", 5)
        self.max_iters = additional.get("max_steps", 3)
        self.model = ExtraTreesClassifier(
            n_estimators=self.trees_in_step,
            criterion=params.get("criterion", "gini"),
            max_features=params.get("max_features", 0.8),
            min_samples_split=params.get("min_samples_split", 4),
            min_samples_leaf=params.get("min_samples_leaf", 4),
            warm_start=True,
            n_jobs=-1,
            random_state=params.get("seed", 1),
        )

    def file_extension(self):
        return "extra_trees"


class ExtraTreesRegressorAlgorithm(SklearnTreesRegressorAlgorithm):

    algorithm_name = "Extra Trees Regressor"
    algorithm_short_name = "Extra Trees"

    def __init__(self, params):
        super(ExtraTreesRegressorAlgorithm, self).__init__(params)
        logger.debug("ExtraTreesRegressorAlgorithm.__init__")

        self.library_version = sklearn.__version__
        self.trees_in_step = additional.get("trees_in_step", 5)
        self.max_iters = additional.get("max_steps", 3)
        self.model = ExtraTreesRegressor(
            n_estimators=self.trees_in_step,
            criterion=params.get("criterion", "mse"),
            max_features=params.get("max_features", 0.8),
            min_samples_split=params.get("min_samples_split", 4),
            min_samples_leaf=params.get("min_samples_leaf", 4),
            warm_start=True,
            n_jobs=-1,
            random_state=params.get("seed", 1),
        )

    def file_extension(self):
        return "extra_trees"


# For binary classification target should be 0, 1. There should be no NaNs in target.
et_params = {
    "criterion": ["gini", "entropy"],
    "max_features": [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0],
    "min_samples_split": [2, 4, 6, 8, 10, 15, 20, 30, 40, 50],
    "min_samples_leaf": range(1, 21),
}

additional = {
    "trees_in_step": 10,
    "train_cant_improve_limit": 5,
    "min_steps": 5,
    "max_steps": 500,
    "max_rows_limit": None,
    "max_cols_limit": None,
}
required_preprocessing = [
    "missing_values_inputation",
    "convert_categorical",
    "target_as_integer",
]

AlgorithmsRegistry.add(
    BINARY_CLASSIFICATION,
    ExtraTreesAlgorithm,
    et_params,
    required_preprocessing,
    additional,
)

AlgorithmsRegistry.add(
    MULTICLASS_CLASSIFICATION,
    ExtraTreesAlgorithm,
    et_params,
    required_preprocessing,
    additional,
)


#
# REGRESSION
#

regression_et_params = {
    "criterion": ["mse"], # remove "mae" because it slows down a lot https://github.com/scikit-learn/scikit-learn/issues/9626
    "max_features": [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0],
    "min_samples_split": [2, 4, 6, 8, 10, 15, 20, 30, 40, 50],
    "min_samples_leaf": range(1, 21),
}

regression_additional = {
    "trees_in_step": 10,
    "train_cant_improve_limit": 5,
    "max_steps": 500,
    "max_rows_limit": None,
    "max_cols_limit": None,
}
regression_required_preprocessing = [
    "missing_values_inputation",
    "convert_categorical",
    "target_scale",
]

AlgorithmsRegistry.add(
    REGRESSION,
    ExtraTreesRegressorAlgorithm,
    regression_et_params,
    regression_required_preprocessing,
    regression_additional,
)
