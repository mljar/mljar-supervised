import logging

log = logging.getLogger(__name__)

from supervised.models.learner_xgboost import XgbLearner

from supervised.models.learner_random_forest import RandomForestLearner
from supervised.models.learner_lightgbm import LightgbmLearner
from supervised.models.learner_catboost import CatBoostLearner
class LearnerFactoryException(Exception):
    def __init__(self, message):
        Exception.__init__(self, message)
        log.error(message)


class LearnerFactory(object):
    @staticmethod
    def get_learner(params):
        learner_type = params.get("model_type", "Xgboost")
        if learner_type == "Xgboost":
            return XgbLearner(params)
        elif learner_type == "RF":
            return RandomForestLearner(params)
        elif learner_type == "LightGBM":
            return LightgbmLearner(params)
        elif learner_type == "CatBoost":
            return CatBoostLearner(params)    
        else:
            msg = "Learner {0} not defined".format(learner_type)
            raise LearnerFactoryException(msg)

    @staticmethod
    def load(json_desc):
        learner = LearnerFactory.get_learner(json_desc.get("params"))
        learner.load(json_desc)
        return learner
