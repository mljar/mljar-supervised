import logging

log = logging.getLogger(__name__)

from supervised.models.learner_xgboost import XgbLearner

from supervised.models.learner_random_forest import RandomForestLearner


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
        if learner_type == "RF":
            return RandomForestLearner(params)
        else:
            msg = "Learner {0} not defined".format(learner_type)
            raise LearnerFactoryException(msg)

    @staticmethod
    def load(json_desc):
        learner = LearnerFactory.get_learner(json_desc.get("params"))
        learner.load(json_desc)
        return learner
