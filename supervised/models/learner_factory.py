import logging

log = logging.getLogger(__name__)

from supervised.models.learner_xgboost import XgbLearner

from supervised.models.learner_random_forest import RandomForestLearner
from supervised.models.learner_lightgbm import LightgbmLearner
from supervised.models.learner_catboost import CatBoostLearner
from supervised.models.learner_nn import NeuralNetworkLearner


class LearnerFactoryException(Exception):
    def __init__(self, message):
        Exception.__init__(self, message)
        log.error(message)


class LearnerFactory(object):

    learners = {
        "Xgboost": XgbLearner,
        "RF": RandomForestLearner,
        "LightGBM": LightgbmLearner,
        "CatBoost": CatBoostLearner,
        "NN": NeuralNetworkLearner,
    }

    @classmethod
    def get_learner(cls, params):
        learner_type = params.get("model_type", "Xgboost")

        if learner_type in cls.learners:
            return cls.learners[learner_type](params)
        else:
            msg = "Learner {0} not defined".format(learner_type)
            raise LearnerFactoryException(msg)

    @classmethod
    def load(cls, json_desc):
        learner = LearnerFactory.get_learner(json_desc.get("params"))
        learner.load(json_desc)
        return learner
