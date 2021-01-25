import logging
import os
import sklearn
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split

from supervised.algorithms.algorithm import BaseAlgorithm
from supervised.algorithms.sklearn import SklearnAlgorithm
from supervised.algorithms.registry import AlgorithmsRegistry
from supervised.algorithms.registry import (
    BINARY_CLASSIFICATION,
    MULTICLASS_CLASSIFICATION,
    REGRESSION,
)
from supervised.utils.config import LOG_LEVEL

logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)


from dtreeviz.trees import dtreeviz


class KNNFit(SklearnAlgorithm):
    def file_extension(self):
        return "k_neighbors"

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
        if X.shape[0] > 1000:
            X1, _, y1, _ = train_test_split(
                X, y, train_size=1000, stratify=y, random_state=1234
            )
            self.model.fit(X1, y1)
        else:
            self.model.fit(X, y)


class KNeighborsAlgorithm(KNNFit):

    algorithm_name = "k-Nearest Neighbors"
    algorithm_short_name = "Nearest Neighbors"

    def __init__(self, params):
        super(KNeighborsAlgorithm, self).__init__(params)
        logger.debug("KNeighborsAlgorithm.__init__")
        self.library_version = sklearn.__version__
        self.max_iters = 1
        self.model = KNeighborsClassifier(
            n_neighbors=params.get("n_neighbors", 3),
            weights=params.get("weights", "uniform"),
            algorithm="kd_tree",
            n_jobs=-1,
        )


class KNeighborsRegressorAlgorithm(KNNFit):

    algorithm_name = "k-Nearest Neighbors"
    algorithm_short_name = "Nearest Neighbors"

    def __init__(self, params):
        super(KNeighborsRegressorAlgorithm, self).__init__(params)
        logger.debug("KNeighborsRegressorAlgorithm.__init__")
        self.library_version = sklearn.__version__
        self.max_iters = 1
        self.model = KNeighborsRegressor(
            n_neighbors=params.get("n_neighbors", 3),
            weights=params.get("weights", "uniform"),
            algorithm="ball_tree",
            n_jobs=-1,
        )


knn_params = {"n_neighbors": [3, 5, 7], "weights": ["uniform", "distance"]}

default_params = {"n_neighbors": 5, "weights": "uniform"}

additional = {"max_rows_limit": 100000, "max_cols_limit": 100}

required_preprocessing = [
    "missing_values_inputation",
    "convert_categorical",
    "datetime_transform",
    "text_transform",
    "scale",
    "target_as_integer",
]

AlgorithmsRegistry.add(
    BINARY_CLASSIFICATION,
    KNeighborsAlgorithm,
    knn_params,
    required_preprocessing,
    additional,
    default_params,
)
AlgorithmsRegistry.add(
    MULTICLASS_CLASSIFICATION,
    KNeighborsAlgorithm,
    knn_params,
    required_preprocessing,
    additional,
    default_params,
)

AlgorithmsRegistry.add(
    REGRESSION,
    KNeighborsRegressorAlgorithm,
    knn_params,
    required_preprocessing,
    additional,
    default_params,
)
