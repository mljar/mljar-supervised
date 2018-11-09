import logging
logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.DEBUG)

import logging
log = logging.getLogger(__name__)

from validation.validation_step import ValidationStep



from callbacks.early_stopping import EarlyStopping
from callbacks.time_constraint import TimeConstraint


from callbacks.callback_list import CallbackList

class LearnerFrameworkParametersException(Exception):
    pass

class LearnerFramework():

    def __init__(self, params, callbacks = []):
        log.debug('LearnerFramework __init__')

        for i in ['learner', 'validation']: # mandatory parameters
            if i not in params:
                msg = 'Missing {0} parameter in LearnerFramework params'.format(i)
                log.error(msg)
                raise ValueError(msg)

        self.params = params
        self.callbacks = CallbackList(callbacks)

        self.preprocessing_params = params.get('preprocessing')
        self.validation_params = params.get('validation')
        self.learner_params = params.get('learner')

        self.validation = None
        self.preprocessings = []
        self.learners = []

    def train(self, data):
        pass

    def predict(self, data):
        pass

    def to_json(self):
        pass

    def from_json(self):
        pass

    def save(self):
        pass

    def load(self):
        pass
