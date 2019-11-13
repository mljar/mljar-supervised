import logging
import os
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

from supervised.algorithms.algorithm import BaseAlgorithm
from supervised.algorithms.sklearn import SklearnTreesClassifierAlgorithm
from supervised.config import storage_path
from supervised.algorithms.registry import AlgorithmsRegistry
from supervised.algorithms.registry import BINARY_CLASSIFICATION
from supervised.algorithms.registry import MULTICLASS_CLASSIFICATION
from supervised.config import LOG_LEVEL

logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)


class RandomForestAlgorithm(SklearnTreesClassifierAlgorithm):

    algorithm_name = "Random Forest"
    algorithm_short_name = "RF"

    def __init__(self, params):
        super(RandomForestAlgorithm, self).__init__(params)
        logger.debug("RandomForestAlgorithm.__init__")

        self.library_version = sklearn.__version__

        self.model_file = self.uid + ".rf.model"
        self.model_file_path = os.path.join(storage_path, self.model_file)

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
    "target_preprocessing",
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
