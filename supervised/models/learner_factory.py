import logging
log = logging.getLogger(__name__)

from .learner_xgboost import XgbLearner

class LearnerFactoryException(Exception):
    def __init__(self, message):
        Exception.__init__(self, message)
        log.error(message)

class LearnerFactory(object):

    @staticmethod
    def get_learner(params):
        learner_type = params.get('learner_type', 'xgb')
        if learner_type == 'xgb':
            return XgbLearner(params)
        else:
            msg = 'Learner {0} not defined'.format(learner_type)
            raise LearnerFactoryException(msg)
