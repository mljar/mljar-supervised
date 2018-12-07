import logging

log = logging.getLogger(__name__)

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold

from supervised.validation.validator_base import BaseValidator


class WithDatasetValidatorException(Exception):
    def __init__(self, message):
        Exception.__init__(self, message)
        log.error(message)


class WithDatasetValidator(BaseValidator):
    def __init__(self, params, data):
        BaseValidator.__init__(self, params, data)

        if self.data.get("validation") is None:
            msg = "Missing validation data"
            raise WithDatasetValidatorException(msg)
        for i in ["X", "y"]:
            if self.data["validation"].get(i) is None:
                msg = "Missing {0} in validation data".format(i)
                raise WithDatasetValidatorException(msg)

    def split(self):
        X_train = self.data["train"]["X"]
        y_train = self.data["train"]["y"]
        X_validation = self.data["validation"]["X"]
        y_validation = self.data["validation"]["y"]

        yield {"X": X_train, "y": y_train}, {"X": X_validation, "y": y_validation}

    def get_n_splits(self):
        return 1
