import logging

log = logging.getLogger(__name__)

from supervised.validation.validator_kfold import KFoldValidator
from supervised.validation.validator_split import SplitValidator
from supervised.validation.validator_with_dataset import WithDatasetValidator


class ValidationStepException(Exception):
    def __init__(self, message):
        Exception.__init__(self, message)
        log.error(message)


class ValidationStep:
    def __init__(self, params):
        
        # kfold is default validation technique
        self.validation_type = params.get("validation_type", "kfold")

        if self.validation_type == "kfold":
            self.validator = KFoldValidator(params)
        else:
            raise Exception("Other validation types are not implemented yet!")
        '''
        elif self.validation_type == "split":
            self.validator = SplitValidator(params, data)
        elif self.validation_type == "with_dataset":
            self.validator = WithDatasetValidator(params, data)
        else:
            msg = "Unknown validation type: {0}".format(self.validation_type)
            raise ValidationStepException(msg)
        '''

    def get_split(self, k):
        return self.validator.get_split(k)

    def split(self):
        return self.validator.split()

    def get_n_splits(self):
        return self.validator.get_n_splits()
