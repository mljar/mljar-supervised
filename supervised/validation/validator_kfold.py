import os
import gc
import logging
import numpy as np
import pandas as pd
import warnings

log = logging.getLogger(__name__)

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold

from supervised.validation.validator_base import BaseValidator
from supervised.exceptions import AutoMLException
from supervised.utils.utils import load_data
from supervised.utils.config import mem
import time


class KFoldValidator(BaseValidator):
    def __init__(self, params):
        BaseValidator.__init__(self, params)

        self.k_folds = self.params.get("k_folds", 5)
        self.shuffle = self.params.get("shuffle", True)
        self.stratify = self.params.get("stratify", False)
        self.random_seed = self.params.get("random_seed", 1906)
        self.repeats = self.params.get("repeats", 1)

        if not self.shuffle and self.repeats > 1:
            warnings.warn("Disable repeats in validation because shuffle is disabled")
            self.repeats = 1

        self.skf = []

        for r in range(self.repeats):
            random_seed = self.random_seed + r if self.shuffle else None
            if self.stratify:
                if self.shuffle:
                    self.skf += [
                        StratifiedKFold(
                            n_splits=self.k_folds,
                            shuffle=self.shuffle,
                            random_state=random_seed,
                        )
                    ]
                else:
                    self.skf += [
                        StratifiedKFold(
                            n_splits=self.k_folds,
                            shuffle=self.shuffle,
                            random_state=random_seed,
                        )
                    ]
            else:
                self.skf += [
                    KFold(
                        n_splits=self.k_folds,
                        shuffle=self.shuffle,
                        random_state=random_seed,
                    )
                ]

        self._results_path = self.params.get("results_path")
        self._X_path = self.params.get("X_path")
        self._y_path = self.params.get("y_path")
        self._sample_weight_path = self.params.get("sample_weight_path")

        if self._X_path is None or self._y_path is None:
            raise AutoMLException("No data path set in KFoldValidator params")

        folds_path = os.path.join(self._results_path, "folds")

        if not os.path.exists(folds_path):

            os.mkdir(folds_path)
            X = load_data(self._X_path)
            y = load_data(self._y_path)
            y = y["target"]

            if isinstance(y[0], bytes):
                # see https://github.com/scikit-learn/scikit-learn/issues/16980
                y = y.astype(str)

            for repeat_cnt, skf in enumerate(self.skf):
                for fold_cnt, (train_index, validation_index) in enumerate(
                    skf.split(X, y)
                ):
                    repeat_str = f"_repeat_{repeat_cnt}" if len(self.skf) > 1 else ""
                    train_index_file = os.path.join(
                        self._results_path,
                        "folds",
                        f"fold_{fold_cnt}{repeat_str}_train_indices.npy",
                    )
                    validation_index_file = os.path.join(
                        self._results_path,
                        "folds",
                        f"fold_{fold_cnt}{repeat_str}_validation_indices.npy",
                    )

                    np.save(train_index_file, train_index)
                    np.save(validation_index_file, validation_index)
            del X
            del y
            gc.collect()

        else:
            log.debug("Folds split already done, reuse it")

    def get_split(self, k, repeat=0):

        repeat_str = f"_repeat_{repeat}" if self.repeats > 1 else ""

        train_index_file = os.path.join(
            self._results_path, "folds", f"fold_{k}{repeat_str}_train_indices.npy"
        )
        validation_index_file = os.path.join(
            self._results_path, "folds", f"fold_{k}{repeat_str}_validation_indices.npy"
        )

        train_index = np.load(train_index_file)
        validation_index = np.load(validation_index_file)

        X = load_data(self._X_path)
        y = load_data(self._y_path)
        y = y["target"]

        sample_weight = None
        if self._sample_weight_path is not None:
            sample_weight = load_data(self._sample_weight_path)
            sample_weight = sample_weight["sample_weight"]

        train_data = {"X": X.loc[train_index], "y": y.loc[train_index]}
        validation_data = {"X": X.loc[validation_index], "y": y.loc[validation_index]}
        if sample_weight is not None:
            train_data["sample_weight"] = sample_weight.loc[train_index]
            validation_data["sample_weight"] = sample_weight.loc[validation_index]

        return (train_data, validation_data)

    def get_n_splits(self):
        return self.k_folds

    def get_repeats(self):
        return self.repeats
