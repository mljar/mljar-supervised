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
    def __init__(self, params, data):
        self.data = data
        self.params = params

        # kfold is default validation technique
        self.validation_type = self.params.get("validation_type", "kfold")

        if self.validation_type == "kfold":
            self.validator = KFoldValidator(params, data)
        elif self.validation_type == "split":
            self.validator = SplitValidator(params, data)
        elif self.validation_type == "with_dataset":
            self.validator = WithDatasetValidator(params, data)
        else:
            msg = "Unknown validation type: {0}".format(self.validation_type)
            raise ValidationStepException(msg)

    def split(self):
        return self.validator.split()

    def get_n_splits(self):
        return self.validator.get_n_splits()
