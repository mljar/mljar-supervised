import logging
import os
import warnings

import numpy as np

log = logging.getLogger(__name__)

from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_consistent_length

from supervised.exceptions import AutoMLException
from supervised.utils.utils import data_row_count, load_data
from supervised.validation.validator_base import BaseValidator


class SplitValidator(BaseValidator):

    def __init__(self, params, data_store=None):
        BaseValidator.__init__(self, params, data_store=data_store)

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
        y = load_data(self._y_path, store=self._data_store)["target"]
        # Keep train_test_split's input-length validation without loading features.
        paths = (self._X_path, self._sample_weight_path, self._sensitive_features_path)
        check_consistent_length(
            y,
            *(
                range(data_row_count(path, store=self._data_store))
                for path in paths
                if path is not None
            ),
        )
        stratify = y if self.stratify and self.shuffle else None
        train_index, validation_index = train_test_split(
            np.arange(len(y)),
            train_size=self.train_ratio,
            test_size=1.0 - self.train_ratio,
            shuffle=self.shuffle,
            stratify=stratify,
            random_state=self.random_seed + repeat,
        )
        train_data = self._load_rows(train_index)
        validation_data = self._load_rows(validation_index)

        repeat_str = f"repeat_{repeat}_" if self.repeats > 1 else ""

        train_data_file = os.path.join(
            self._results_path, f"split_{repeat_str}train_indices.npy"
        )
        validation_data_file = os.path.join(
            self._results_path, f"split_{repeat_str}validation_indices.npy"
        )

        np.save(train_data_file, train_data["X"].index)
        np.save(validation_data_file, validation_data["X"].index)

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
from sklearn.utils.validation import check_consistent_length

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
