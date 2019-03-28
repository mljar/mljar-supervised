import logging
import copy
import numpy as np
import pandas as pd

from supervised.models.learner import Learner
from supervised.tuner.registry import ModelsRegistry
from supervised.tuner.registry import BINARY_CLASSIFICATION

import operator

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.models import model_from_json

log = logging.getLogger(__name__)


class NeuralNetworkLearner(Learner):

    algorithm_name = "Neural Network"
    algorithm_short_name = "NN"

    def __init__(self, params):
        super(NeuralNetworkLearner, self).__init__(params)
        self.library_version = keras.__version__
        self.model_file = self.uid + ".nn.model"
        self.model_file_path = "/tmp/" + self.model_file

        self.rounds = additional.get("one_step", 50)
        self.max_iters = additional.get("max_steps", 3)
        self.learner_params = {}
        self.model = Sequential()
        self.model.add(Dense(32, activation="relu", input_dim=6))
        self.model.add(Dense(1, activation="sigmoid"))
        self.model.compile(
            optimizer="sgd", loss="binary_crossentropy", metrics=["accuracy"]
        )

        log.debug("NeuralNetworkLearner __init__")

    def update(self, update_params):
        print("NN update", update_params)
        self.rounds = update_params["step"]

    def fit(self, data):
        log.debug("NNLearner.fit")
        X = data.get("X")
        y = data.get("y")
        print(X.head(5))
        self.model.fit(X, y, batch_size=256, nb_epoch=1)

    def predict(self, X):
        print("Predict", np.unique(np.ravel(self.model.predict(X))))
        return np.ravel(self.model.predict(X))

    def copy(self):
        return None

    def save(self):

        self.model.save_weights(self.model_file_path)

        json_desc = {
            "library_version": self.library_version,
            "algorithm_name": self.algorithm_name,
            "algorithm_short_name": self.algorithm_short_name,
            "uid": self.uid,
            "model_file": self.model_file,
            "model_file_path": self.model_file_path,
            "params": self.params,
            "model_architecture_json": self.model.to_json(),
        }

        log.debug("NeuralNetworkLearner save model to %s" % self.model_file_path)
        return json_desc

    def load(self, json_desc):

        self.library_version = json_desc.get("library_version", self.library_version)
        self.algorithm_name = json_desc.get("algorithm_name", self.algorithm_name)
        self.algorithm_short_name = json_desc.get(
            "algorithm_short_name", self.algorithm_short_name
        )
        self.uid = json_desc.get("uid", self.uid)
        self.model_file = json_desc.get("model_file", self.model_file)
        self.model_file_path = json_desc.get("model_file_path", self.model_file_path)
        self.params = json_desc.get("params", self.params)
        model_json = json_desc.get("model_architecture_json")

        log.debug("NeuralNetworkLearner load model from %s" % self.model_file_path)

        self.model = model_from_json(model_json)
        self.model.load_weights(self.model_file_path)
        print("Loaded model from disk")

    def importance(self, column_names, normalize=True):
        return None


NeuralNetworkLearnerBinaryClassificationParams = {
    "dense_layers": [1, 2, 3],
    "dense_1_size": [5, 10, 20, 50, 100],
    "dense_2_size": [5, 10, 20, 50, 100],
    "dense_3_size": [5, 10, 20, 50, 100],
    "dense_4_size": [5, 10, 20, 50, 100],
    "dense_5_size": [5, 10, 20, 50, 100],
    "dense_6_size": [5, 10, 20, 50, 100],
    "optimize": ["adadelta", "sgd"],  #'sgd',
    "activation": ["relu", "prelu", "leakyrelu"],
    "dropout": [0, 0.25, 0.5],
}

additional = {
    "one_step": 1,
    "train_cant_improve_limit": 5,
    "max_steps": 25,
    "max_rows_limit": None,
    "max_cols_limit": None,
}
required_preprocessing = [
    "missing_values_inputation",
    "convert_categorical",
    "scale",
    "target_preprocessing",
]

ModelsRegistry.add(
    BINARY_CLASSIFICATION,
    NeuralNetworkLearner,
    NeuralNetworkLearnerBinaryClassificationParams,
    required_preprocessing,
    additional,
)
