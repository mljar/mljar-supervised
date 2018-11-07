import logging
log = logging.getLogger(__name__)


from validator_kfold import KFoldValidator
from validator_split import SplitValidator
from validator_with_dataset import WithDatasetValidator

class ValidationStepException(Exception):
    def __init__(self, message):
        Exception.__init__(self, message)
        log.error(message)

class ValidationStep():

    def __init__(self, data, params):
        self.data = data
        self.params = params

        # kfold is default validation technique
        self.validator_type = self.params.get('validator_type', 'kfold')

        if self.validator_type == 'kfold':
            self.validator = KFoldValidator(data, params)
        elif self.validator_type == 'split':
            self.validator = SplitValidator(data, params)
        elif self.validator_type == 'split':
            self.validator = WithDatasetValidator(data, params)
        else:
            msg = 'Unknown validation type: {0}'.format(self.validator_type)
            raise ValidationStepException(msg)

    def split(self):
        return self.validator.split()

    def get_n_splits(self):
        return self.validator.get_n_splits()
