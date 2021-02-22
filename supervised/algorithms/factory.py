from supervised.algorithms.registry import AlgorithmsRegistry, BINARY_CLASSIFICATION

import logging

logger = logging.getLogger(__name__)

from supervised.exceptions import AutoMLException


class AlgorithmFactory(object):
    @classmethod
    def get_algorithm(cls, params):
        alg_type = params.get("model_type", "Xgboost")
        ml_task = params.get("ml_task", BINARY_CLASSIFICATION)

        try:
            Algorithm = AlgorithmsRegistry.get_algorithm_class(ml_task, alg_type)
            return Algorithm(params)
        except Exception as e:
            raise AutoMLException(f"Cannot get algorithm class. {str(e)}")

    @classmethod
    def load(cls, json_desc, learner_path, lazy_load):
        learner = AlgorithmFactory.get_algorithm(json_desc.get("params"))
        learner.set_params(json_desc, learner_path)
        if not lazy_load:
            learner.reload()
        return learner
