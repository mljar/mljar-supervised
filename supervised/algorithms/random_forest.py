import logging
import os
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

from supervised.algorithms.algorithm import BaseAlgorithm
from supervised.algorithms.sklearn import SklearnTreesClassifierAlgorithm
from supervised.algorithms.sklearn import SklearnTreesRegressorAlgorithm
from supervised.algorithms.registry import AlgorithmsRegistry
from supervised.algorithms.registry import BINARY_CLASSIFICATION
from supervised.algorithms.registry import MULTICLASS_CLASSIFICATION
from supervised.algorithms.registry import REGRESSION
from supervised.utils.config import storage_path
from supervised.utils.config import LOG_LEVEL

logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)


class RandomForestAlgorithm(SklearnTreesClassifierAlgorithm):

    algorithm_name = "Random Forest"
    algorithm_short_name = "Random Forest"

    def __init__(self, params):
        super(RandomForestAlgorithm, self).__init__(params)
        logger.debug("RandomForestAlgorithm.__init__")

        self.library_version = sklearn.__version__
        self.trees_in_step = additional.get("trees_in_step", 5)
        self.max_iters = additional.get("max_steps", 3)
        self.model = RandomForestClassifier(
            n_estimators=self.trees_in_step,
            criterion=params.get("criterion", "gini"),
            max_features=params.get("max_features", 0.8),
            min_samples_split=params.get("min_samples_split", 4),
            min_samples_leaf=params.get("min_samples_leaf", 4),
            warm_start=True,
            n_jobs=-1,
            random_state=params.get("seed", 1),
        )

    def file_extenstion(self):
        return "random_forest"


class RandomForestRegressorAlgorithm(SklearnTreesRegressorAlgorithm):

    algorithm_name = "Random Forest Regressor"
    algorithm_short_name = "Random Forest Regressor"

    def __init__(self, params):
        super(RandomForestRegressorAlgorithm, self).__init__(params)
        logger.debug("RandomForestRegressorAlgorithm.__init__")

        self.library_version = sklearn.__version__
        self.trees_in_step = additional.get("trees_in_step", 5)
        self.max_iters = additional.get("max_steps", 3)
        self.model = RandomForestRegressor(
            n_estimators=self.trees_in_step,
            criterion=params.get("criterion", "gini"),
            max_features=params.get("max_features", 0.8),
            min_samples_split=params.get("min_samples_split", 4),
            min_samples_leaf=params.get("min_samples_leaf", 4),
            warm_start=True,
            n_jobs=-1,
            random_state=params.get("seed", 1),
        )

    def file_extenstion(self):
        return "random_forest"


# For binary classification target should be 0, 1. There should be no NaNs in target.
rf_params = {
    "criterion": ["gini", "entropy"],
    "max_features": [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0],
    "min_samples_split": [2, 4, 6, 8, 10, 15, 20, 30, 40, 50],
    "min_samples_leaf": range(1, 21),
}

additional = {
    "trees_in_step": 10,
    "train_cant_improve_limit": 5,
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
    RandomForestAlgorithm,
    rf_params,
    required_preprocessing,
    additional,
)

AlgorithmsRegistry.add(
    MULTICLASS_CLASSIFICATION,
    RandomForestAlgorithm,
    rf_params,
    required_preprocessing,
    additional,
)


#
# REGRESSION
#

regression_rf_params = {
    "criterion": ["mse", "mae"],
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
regression_required_preprocessing = ["missing_values_inputation", "convert_categorical"]

AlgorithmsRegistry.add(
    REGRESSION,
    RandomForestRegressorAlgorithm,
    regression_rf_params,
    regression_required_preprocessing,
    regression_additional,
)
