import logging

log = logging.getLogger(__name__)

from sklearn.model_selection import train_test_split

from supervised.validation.validator_base import BaseValidator


class SplitValidatorException(Exception):
    def __init__(self, message):
        Exception.__init__(self, message)
        log.error(message)


class SplitValidator(BaseValidator):
    def __init__(self, params, data):
        BaseValidator.__init__(self, params, data)

        self.train_ratio = self.params.get("train_ratio", 0.8)
        self.shuffle = self.params.get("shuffle", True)
        self.stratify = self.params.get("stratify", False)
        self.random_seed = self.params.get("random_seed", 1706)
        log.debug("SplitValidator, train_ratio: {0}".format(self.train_ratio))

    def split(self):
        X = self.data["train"]["X"]
        y = self.data["train"]["y"]

        X_train, X_validation, y_train, y_validation = train_test_split(
            X,
            y,
            train_size=self.train_ratio,
            test_size=1.0 - self.train_ratio,
            stratify=y if self.stratify else None,
            random_state=self.random_seed,
        )
        yield {"X": X_train, "y": y_train}, {"X": X_validation, "y": y_validation}

    def get_n_splits(self):
        return 1


"""
import numpy as np
import pandas as pd

from sklearn.utils.fixes import bincount
from sklearn.model_selection import train_test_split

import logging
logger = logging.getLogger('mljar')


def validation_split(train, validation_train_split, stratify, shuffle, random_seed):

    if shuffle:
    else:
        if stratify is None:
            train, validation = data_split(validation_train_split, train)
        else:
            train, validation = data_split_stratified(validation_train_split, train, stratify)
    return train, validation


"""
