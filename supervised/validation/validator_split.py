import os
import gc
import logging
import numpy as np
import pandas as pd
import warnings

log = logging.getLogger(__name__)

from sklearn.model_selection import train_test_split
from supervised.validation.validator_base import BaseValidator
from supervised.exceptions import AutoMLException

from supervised.utils.config import mem
import time


class SplitValidator(BaseValidator):
    def __init__(self, params):
        BaseValidator.__init__(self, params)

        self.train_ratio = self.params.get("train_ratio", 0.8)
        self.shuffle = self.params.get("shuffle", True)
        self.stratify = self.params.get("stratify", False)
        self.random_seed = self.params.get("random_seed", 1234)
        self.repeats = self.params.get("repeats", 1)

        if not self.shuffle and self.repeats > 1:
            warnings.warn("Disable repeats in validation because shuffle is disabled")
            self.repeats = 1

        self._results_path = self.params.get("results_path")
        self._X_path = self.params.get("X_path")
        self._y_path = self.params.get("y_path")
        self._sample_weight_path = self.params.get("sample_weight_path")

        if self._X_path is None or self._y_path is None:
            raise AutoMLException("No data path set in SplitValidator params")

    def get_split(self, k=0, repeat=0):

        X = pd.read_parquet(self._X_path)
        y = pd.read_parquet(self._y_path)
        y = y["target"]

        sample_weight = None
        if self._sample_weight_path is not None:
            sample_weight = pd.read_parquet(self._sample_weight_path)
            sample_weight = sample_weight["sample_weight"]

        stratify = None
        if self.stratify:
            stratify = y
        if self.shuffle == False:
            stratify = None

        if sample_weight is not None:
            X_train, X_validation, y_train, y_validation, sample_weight_train, sample_weight_validation = train_test_split(
                X,
                y,
                sample_weight,
                train_size=self.train_ratio,
                test_size=1.0 - self.train_ratio,
                shuffle=self.shuffle,
                stratify=stratify,
                random_state=self.random_seed + repeat,
            )
        else:
            X_train, X_validation, y_train, y_validation = train_test_split(
                X,
                y,
                train_size=self.train_ratio,
                test_size=1.0 - self.train_ratio,
                shuffle=self.shuffle,
                stratify=stratify,
                random_state=self.random_seed + repeat,
            )
        train_data = {"X": X_train, "y": y_train}
        validation_data = {"X": X_validation, "y": y_validation}
        if sample_weight is not None:
            train_data["sample_weight"] = sample_weight_train
            validation_data["sample_weight"] = sample_weight_validation

        return train_data, validation_data

    def get_n_splits(self):
        return 1

    def get_repeats(self):
        return self.repeats


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
