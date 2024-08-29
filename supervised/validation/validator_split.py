import logging
import os
import warnings

import numpy as np

log = logging.getLogger(__name__)

from sklearn.model_selection import train_test_split

from supervised.exceptions import AutoMLException
from supervised.utils.utils import load_data
from supervised.validation.validator_base import BaseValidator


class SplitValidator(BaseValidator):
    def __init__(self, params):
        BaseValidator.__init__(self, params)

        self.train_ratio = self.params.get("train_ratio", 0.8)
        self.shuffle = self.params.get("shuffle", True)
        self.stratify = self.params.get("stratify", False)
        self.random_seed = self.params.get("random_seed", 1234)
        self.repeats = self.params.get("repeats", 1)

        if not self.shuffle and self.repeats > 1:
            warnings.warn(
                "Disable repeats in validation because shuffle is disabled", UserWarning
            )
            self.repeats = 1

        self._results_path = self.params.get("results_path")
        self._X_path = self.params.get("X_path")
        self._y_path = self.params.get("y_path")
        self._sample_weight_path = self.params.get("sample_weight_path")
        self._sensitive_features_path = self.params.get("sensitive_features_path")

        if self._X_path is None or self._y_path is None:
            raise AutoMLException("No data path set in SplitValidator params")

    def get_split(self, k=0, repeat=0):
        X = load_data(self._X_path)
        y = load_data(self._y_path)
        y = y["target"]

        sample_weight = None
        if self._sample_weight_path is not None:
            sample_weight = load_data(self._sample_weight_path)
            sample_weight = sample_weight["sample_weight"]

        sensitive_features = None
        if self._sensitive_features_path is not None:
            sensitive_features = load_data(self._sensitive_features_path)

        stratify = None
        if self.stratify:
            stratify = y
        if self.shuffle == False:
            stratify = None

        input_data = [X, y]
        if sample_weight is not None:
            input_data += [sample_weight]
        if sensitive_features is not None:
            input_data += [sensitive_features]

        output_data = train_test_split(
            *input_data,
            train_size=self.train_ratio,
            test_size=1.0 - self.train_ratio,
            shuffle=self.shuffle,
            stratify=stratify,
            random_state=self.random_seed + repeat,
        )

        X_train = output_data[0]
        X_validation = output_data[1]
        y_train = output_data[2]
        y_validation = output_data[3]
        if sample_weight is not None:
            sample_weight_train = output_data[4]
            sample_weight_validation = output_data[5]
            if sensitive_features is not None:
                sensitive_features_train = output_data[6]
                sensitive_features_validation = output_data[7]
        else:
            if sensitive_features is not None:
                sensitive_features_train = output_data[4]
                sensitive_features_validation = output_data[5]

        train_data = {"X": X_train, "y": y_train}
        validation_data = {"X": X_validation, "y": y_validation}
        if sample_weight is not None:
            train_data["sample_weight"] = sample_weight_train
            validation_data["sample_weight"] = sample_weight_validation
        if sensitive_features is not None:
            train_data["sensitive_features"] = sensitive_features_train
            validation_data["sensitive_features"] = sensitive_features_validation

        repeat_str = f"repeat_{repeat}_" if self.repeats > 1 else ""

        train_data_file = os.path.join(
            self._results_path, f"split_{repeat_str}train_indices.npy"
        )
        validation_data_file = os.path.join(
            self._results_path, f"split_{repeat_str}validation_indices.npy"
        )

        np.save(train_data_file, X_train.index)
        np.save(validation_data_file, X_validation.index)

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
