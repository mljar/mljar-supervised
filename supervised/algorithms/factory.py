from supervised.algorithms.xgboost import XgbAlgorithm
from supervised.algorithms.random_forest import RandomForestAlgorithm
from supervised.algorithms.random_forest import RandomForestRegressorAlgorithm
from supervised.algorithms.lightgbm import LightgbmAlgorithm
from supervised.algorithms.catboost import CatBoostAlgorithm
from supervised.algorithms.nn import NeuralNetworkAlgorithm

from supervised.algorithms.registry import AlgorithmsRegistry, BINARY_CLASSIFICATION

import logging

logger = logging.getLogger(__name__)

from supervised.exceptions import AutoMLException


class AlgorithmFactory(object):

    # algorithms = {
    #    "Xgboost": XgbAlgorithm,
    #    "Random Forest": RandomForestAlgorithm,
    #    "Random Forest Regressor": RandomForestRegressorAlgorithm,
    #    "LightGBM": LightgbmAlgorithm,
    #    "CatBoost": CatBoostAlgorithm,
    #    "NN": NeuralNetworkAlgorithm,
    # }

    @classmethod
    def get_algorithm(cls, params):

        print(params)
        alg_type = params.get("model_type", "Xgboost")
        ml_task = params.get("ml_task", BINARY_CLASSIFICATION)

        try:
            Algorithm = AlgorithmsRegistry.get_algorithm_class(ml_task, alg_type)
            return Algorithm(params)
        except Exception as e:
            raise AutoMLException(f"Cannot get algorithm class. {str(e)}")

        # if alg_type in cls.algorithms:
        #    return cls.algorithms[alg_type](params)
        # else:
        #    msg = "Algorithm {0} not defined".format(alg_type)
        #    raise AlgorithmFactoryException(msg)

    @classmethod
    def load(cls, json_desc, model_file_path):
        learner = AlgorithmFactory.get_algorithm(json_desc.get("params"))
        learner.load(model_file_path)
        return learner
