from supervised.algorithms.xgboost import XgbAlgorithm

from supervised.algorithms.random_forest import RandomForestAlgorithm
from supervised.algorithms.lightgbm import LightgbmAlgorithm
from supervised.algorithms.catboost import CatBoostAlgorithm
from supervised.algorithms.nn import NeuralNetworkAlgorithm


import logging

logger = logging.getLogger(__name__)
# logger.setLevel(logging.ERROR)


class AlgorithmFactoryException(Exception):
    def __init__(self, message):
        super(AlgorithmFactoryException, self).__init__(message)
        logger.error(message)


class AlgorithmFactory(object):

    algorithms = {
        "Xgboost": XgbAlgorithm,
        "RF": RandomForestAlgorithm,
        "LightGBM": LightgbmAlgorithm,
        "CatBoost": CatBoostAlgorithm,
        "NN": NeuralNetworkAlgorithm,
    }

    @classmethod
    def get_algorithm(cls, params):
        alg_type = params.get("model_type", "Xgboost")

        if alg_type in cls.algorithms:
            return cls.algorithms[alg_type](params)
        else:
            msg = "Algorithm {0} not defined".format(alg_type)
            raise LearnerFactoryException(msg)

    @classmethod
    def load(cls, json_desc):
        learner = AlgorithmFactory.get_algorithm(json_desc.get("params"))
        learner.load(json_desc)
        return learner
