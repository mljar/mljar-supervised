import os
import gc
import logging
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold

from supervised.validation.validator_base import BaseValidator
from supervised.exceptions import AutoMLException

from supervised.utils.config import mem
import time


class KFoldValidator(BaseValidator):
    def __init__(self, params):
        BaseValidator.__init__(self, params)

        self.k_folds = self.params.get("k_folds", 5)
        self.shuffle = self.params.get("shuffle", True)
        self.stratify = self.params.get("stratify", False)
        self.random_seed = self.params.get("random_seed", 1906)

        if self.stratify:
            if self.shuffle:
                self.skf = StratifiedKFold(
                    n_splits=self.k_folds,
                    shuffle=self.shuffle,
                    random_state=self.random_seed if self.shuffle else None,
                )
            else:
                self.skf = StratifiedKFold(n_splits=self.k_folds, shuffle=self.shuffle)
        else:
            self.skf = KFold(
                n_splits=self.k_folds,
                shuffle=self.shuffle,
                random_state=self.random_seed if self.shuffle else None,
            )

        self._results_path = self.params.get("results_path")
        self._X_train_path = self.params.get("X_train_path")
        self._y_train_path = self.params.get("y_train_path")

        if self._X_train_path is None or self._y_train_path is None:
            raise AutoMLException("No training data path set in KFoldValidator params")

        folds_path = os.path.join(self._results_path, "folds")

        if not os.path.exists(folds_path):

            os.mkdir(folds_path)

            X = pd.read_parquet(self._X_train_path)
            y = pd.read_parquet(self._y_train_path)
            y = y["target"]

            if isinstance(y[0], bytes):
                # see https://github.com/scikit-learn/scikit-learn/issues/16980
                y = y.astype(str)

            for fold_cnt, (train_index, validation_index) in enumerate(
                self.skf.split(X, y)
            ):

                train_index_file = os.path.join(
                    self._results_path, "folds", f"fold_{fold_cnt}_train_indices.npy"
                )
                validation_index_file = os.path.join(
                    self._results_path,
                    "folds",
                    f"fold_{fold_cnt}_validation_indices.npy",
                )

                np.save(train_index_file, train_index)
                np.save(validation_index_file, validation_index)

            del X
            del y
            gc.collect()

        else:
            log.debug("Folds split already done, reuse it")

    def get_split(self, k):

        train_index_file = os.path.join(
            self._results_path, "folds", f"fold_{k}_train_indices.npy"
        )
        validation_index_file = os.path.join(
            self._results_path, "folds", f"fold_{k}_validation_indices.npy"
        )

        train_index = np.load(train_index_file)
        validation_index = np.load(validation_index_file)

        X = pd.read_parquet(self._X_train_path)
        y = pd.read_parquet(self._y_train_path)
        y = y["target"]

        return (
            {"X": X.loc[train_index], "y": y.loc[train_index]},
            {"X": X.loc[validation_index], "y": y.loc[validation_index]},
        )

    def get_n_splits(self):
        return self.k_folds
