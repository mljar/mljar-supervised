import logging

log = logging.getLogger(__name__)

from supervised.validation.validator_kfold import KFoldValidator
from supervised.validation.validator_split import SplitValidator
from supervised.validation.validator_with_dataset import WithDatasetValidator

from supervised.exceptions import AutoMLException


class ValidationStep:
    def __init__(self, params):

        # kfold is default validation technique
        self.validation_type = params.get("validation_type", "kfold")

        if self.validation_type == "kfold":
            self.validator = KFoldValidator(params)
        elif self.validation_type == "split":
            self.validator = SplitValidator(params)
        else:
            raise AutoMLException(
                f"The validation type ({self.validation_type}) is not implemented."
            )
        """
        
        elif self.validation_type == "with_dataset":
            self.validator = WithDatasetValidator(params, data)
        else:
            msg = "Unknown validation type: {0}".format(self.validation_type)
            raise ValidationStepException(msg)
        """

    def get_split(self, k, repeat=0):
        return self.validator.get_split(k, repeat)

    def split(self):
        return self.validator.split()

    def get_n_splits(self):
        return self.validator.get_n_splits()

    def get_repeats(self):
        return self.validator.get_repeats()
