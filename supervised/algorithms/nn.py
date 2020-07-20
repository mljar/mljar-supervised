################################################################################
# switch off tf warnings
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import os
import sys

stderr = sys.stderr
sys.stderr = open(os.devnull, "w")
import keras

sys.stderr = stderr
################################################################################
# set seed for reproducibility
# however in case of running on multiple CPU or GPU there is no reproducibility
import numpy as np
import tensorflow as tf
import random as rn

np.random.seed(42)
rn.seed(12345)
from keras import backend as K

tf.random.set_seed(1234)
# tf.logging.set_verbosity(tf.logging.ERROR)
################################################################################
import logging
import copy
import numpy as np
import pandas as pd
import os
import json
import keras

from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.models import model_from_json
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.callbacks import EarlyStopping

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


class NeuralNetworkAlgorithm(BaseAlgorithm):

    algorithm_name = "Neural Network"
    algorithm_short_name = "Neural Network"

    def __init__(self, params):
        super(NeuralNetworkAlgorithm, self).__init__(params)

        self.library_version = keras.__version__

        self.rounds = additional.get("one_step", 1)
        self.max_iters = additional.get("max_steps", 1)
        self.learner_params = {
            "dense_layers": params.get("dense_layers"),
            "dense_1_size": params.get("dense_1_size"),
            "dense_2_size": params.get("dense_2_size"),
            "dropout": params.get("dropout"),
            "learning_rate": params.get("learning_rate"),
            "momentum": params.get("momentum"),
            "decay": params.get("decay"),
        }
        self.model = None  # we need input data shape to construct model

        if "model_architecture_json" in params:
            self.model = model_from_json(
                json.loads(params.get("model_architecture_json"))
            )
            self.compile_model()

        logger.debug("NeuralNetworkAlgorithm __init__")

    def create_model(self, input_dim):
        self.model = Sequential()
        for i in range(self.learner_params.get("dense_layers")):
            self.model.add(
                Dense(
                    self.learner_params.get("dense_{}_size".format(i + 1)),
                    activation="relu",
                    input_dim=input_dim,
                )
            )
            if (
                self.learner_params.get("dropout") is not None
                and self.learner_params.get("dropout") > 0
            ):
                self.model.add(Dropout(rate=self.learner_params.get("dropout")))

        if self.ml_task == MULTICLASS_CLASSIFICATION:
            self.model.add(Dense(self.params["num_class"], activation="softmax"))
        elif self.ml_task == BINARY_CLASSIFICATION:
            self.model.add(Dense(1, activation="sigmoid"))
        else:
            self.model.add(Dense(1))

        self.compile_model()

    def compile_model(self):
        sgd_opt = SGD(
            lr=self.learner_params.get("learning_rate"),
            momentum=self.learner_params.get("momentum"),
            decay=self.learner_params.get("decay"),
            nesterov=True,
        )

        if self.ml_task == MULTICLASS_CLASSIFICATION:
            self.model.compile(optimizer=sgd_opt, loss="categorical_crossentropy")
        elif self.ml_task == BINARY_CLASSIFICATION:
            self.model.compile(optimizer=sgd_opt, loss="binary_crossentropy")
        else:
            self.model.compile(optimizer=sgd_opt, loss="mean_squared_error")

    def update(self, update_params):
        pass

    def fit(self, X, y, X_validation=None, y_validation=None, log_to_file=None):

        if self.model is None:
            self.create_model(input_dim=X.shape[1])

        batch_size = 1024
        if X.shape[0] < batch_size * 5:
            batch_size = 32

        self.model.fit(X, y, batch_size=batch_size, epochs=self.rounds, verbose=False)

        """
        # Experimental ...
        es = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=50)
        mc = ModelCheckpoint(
            "best_model.h5",
            monitor="val_loss",
            mode="min",
            verbose=0,
            save_best_only=True,
        )
        self.model.fit(
            X,
            y,
            validation_data=(X_validation, y_validation),
            batch_size=4096,
            epochs=1000,
            verbose=False,
            callbacks=[es, mc],
        )
        self.model = load_model("best_model.h5")
        """

    def predict(self, X):
        if "num_class" in self.params:
            return self.model.predict(X)
        return np.ravel(self.model.predict(X))

    def copy(self):
        return copy.deepcopy(self)

    def save(self, model_file_path):
        self.model.save_weights(model_file_path)
        logger.debug(f"Neural Network save model to {model_file_path}")

    def load(self, model_file_path):
        logger.debug(f"Load Neural Network from {model_file_path}")
        self.model.load_weights(model_file_path)

    def get_params(self):
        self.params["model_architecture_json"] = json.dumps(
            self.model.to_json(), indent=4
        )
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
        model_json = self.params.get("model_architecture_json")
        if model_json is not None and self.model is None:
            self.model = model_from_json(json.loads(model_json))
            self.compile_model()

    def file_extension(self):
        return "neural_network"


nn_params = {
    "dense_layers": [2],
    "dense_1_size": [16, 32, 64, 128],
    "dense_2_size": [4, 8, 16, 32],
    "dropout": [0, 0.1, 0.25],
    "learning_rate": [0.01, 0.05, 0.08, 0.1],
    "momentum": [0.85, 0.9, 0.95],
    "decay": [0.0001, 0.001, 0.01],
}

default_nn_params = {
    "dense_layers": 2,
    "dense_1_size": 32,
    "dense_2_size": 16,
    "dropout": 0,
    "learning_rate": 0.05,
    "momentum": 0.9,
    "decay": 0.001,
}

additional = {
    "one_step": 10,
    "train_cant_improve_limit": 5,
    "max_steps": 500,
    "min_steps": 5,
    "max_rows_limit": None,
    "max_cols_limit": None,
}

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
    NeuralNetworkAlgorithm,
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
    "target_as_one_hot",
]
AlgorithmsRegistry.add(
    MULTICLASS_CLASSIFICATION,
    NeuralNetworkAlgorithm,
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
    NeuralNetworkAlgorithm,
    nn_params,
    required_preprocessing,
    additional,
    default_nn_params,
)
