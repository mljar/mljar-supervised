import numpy as np
import pandas as pd
import warnings
import logging

from supervised.algorithms.algorithm import BaseAlgorithm
from supervised.algorithms.sklearn import SklearnAlgorithm
from supervised.algorithms.registry import AlgorithmsRegistry
from supervised.algorithms.registry import (
    BINARY_CLASSIFICATION,
    MULTICLASS_CLASSIFICATION,
    REGRESSION,
)

import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor

from supervised.utils.config import LOG_LEVEL

logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)


class NNFit(SklearnAlgorithm):
    def file_extension(self):
        return "neural_network"

    def is_fitted(self):
        return (
            hasattr(self.model, "n_iter_")
            and self.model.n_iter_ is not None
            and self.model.n_iter_ > 0
        )

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
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore")
            # filter 
            # X does not have valid feature names, but MLPClassifier was fitted with feature names
            self.model.fit(X, y)
        
        if log_to_file is not None:
            loss_curve = self.model.loss_curve_
            result = pd.DataFrame(
                {
                    "iteration": range(len(loss_curve)),
                    "train": loss_curve,
                    "validation": None,
                }
            )
            result.to_csv(log_to_file, index=False, header=False)


class MLPAlgorithm(NNFit):

    algorithm_name = "Neural Network"
    algorithm_short_name = "Neural Network"

    def __init__(self, params):
        super(MLPAlgorithm, self).__init__(params)
        logger.debug("MLPAlgorithm.__init__")
        self.max_iters = 1
        self.library_version = sklearn.__version__
        h1 = params.get("dense_1_size", 32)
        h2 = params.get("dense_2_size", 16)
        learning_rate = params.get("learning_rate", 0.05)

        max_iter = 500
        self.model = MLPClassifier(
            hidden_layer_sizes=(h1, h2),
            activation="relu",
            solver="adam",
            learning_rate=params.get("learning_rate_type", "constant"),
            learning_rate_init=learning_rate,
            alpha=params.get("alpha", 0.0001),
            early_stopping=True,
            n_iter_no_change=50,
            max_iter=max_iter,
            random_state=params.get("seed", 123),
        )

    def get_metric_name(self):
        return "logloss"


class MLPRegressorAlgorithm(NNFit):

    algorithm_name = "Neural Network"
    algorithm_short_name = "Neural Network"

    def __init__(self, params):
        super(MLPRegressorAlgorithm, self).__init__(params)
        logger.debug("MLPRegressorAlgorithm.__init__")
        self.max_iters = 1
        self.library_version = sklearn.__version__
        h1 = params.get("dense_1_size", 32)
        h2 = params.get("dense_2_size", 16)
        learning_rate = params.get("learning_rate", 0.05)
        momentum = params.get("momentum", 0.9)
        early_stopping = True
        max_iter = 500
        self.model = MLPRegressor(
            hidden_layer_sizes=(h1, h2),
            activation="relu",
            solver="adam",
            learning_rate="constant",
            learning_rate_init=learning_rate,
            momentum=momentum,
            early_stopping=early_stopping,
            max_iter=max_iter,
        )

    def get_metric_name(self):
        return "mse"


nn_params = {
    "dense_1_size": [16, 32, 64],
    "dense_2_size": [4, 8, 16, 32],
    "learning_rate": [0.01, 0.05, 0.08, 0.1],
}

default_nn_params = {"dense_1_size": 32, "dense_2_size": 16, "learning_rate": 0.05}

additional = {"max_rows_limit": None, "max_cols_limit": None}

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
    MLPAlgorithm,
    nn_params,
    required_preprocessing,
    additional,
    default_nn_params,
)

AlgorithmsRegistry.add(
    MULTICLASS_CLASSIFICATION,
    MLPAlgorithm,
    nn_params,
    required_preprocessing,
    additional,
    default_nn_params,
)

required_preprocessing = [
    "missing_values_inputation",
    "convert_categorical",
    "datetime_transform",
    "text_transform",
    "scale",
    "target_scale",
]

AlgorithmsRegistry.add(
    REGRESSION,
    MLPRegressorAlgorithm,
    nn_params,
    required_preprocessing,
    additional,
    default_nn_params,
)
