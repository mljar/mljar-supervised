import logging
logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.DEBUG)

import logging
log = logging.getLogger(__name__)

from validation.validation_step import ValidationStep


class LearnerFrameworkParametersException(Exception):
    pass

class LearnerFramework():

    def __init__(self, params, callbacks = None):
        log.debug('LearnerFramework __init__')

        for i in ['model', 'metrics']: # mandatory parameters
            if i not in params:
                msg = 'Missing {0} parameter in train_params'.format(i)
                log.error(msg)
                raise LearnerFrameworkParametersException(msg)

        self.preprocessing = train_params.get('preprocessing')
        self.validation = ValidationStep(train_params.get('validation'))
        self.model = train_params.get('model')

    def train(self, data):
        print('--- LearnerFramework start train ---')

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
