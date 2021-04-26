import os
import gc
import joblib
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


class CustomValidator(BaseValidator):
    def __init__(self, params):
        BaseValidator.__init__(self, params)

        cv_path = self.params.get("cv_path")

        if cv_path is None:
            raise AutoMLException("You need to specify `cv` as list or iterable")

        self.cv = joblib.load(cv_path)
        self.cv = list(self.cv)

        self._results_path = self.params.get("results_path")
        self._X_path = self.params.get("X_path")
        self._y_path = self.params.get("y_path")
        self._sample_weight_path = self.params.get("sample_weight_path")

        if self._X_path is None or self._y_path is None:
            raise AutoMLException("No data path set in CustomValidator params")

        folds_path = os.path.join(self._results_path, "folds")

        if not os.path.exists(folds_path):

            os.mkdir(folds_path)

            print("Custom validation strategy")
            for fold_cnt, (train_index, validation_index) in enumerate(self.cv):

                print(f"Split {fold_cnt}.")
                print(f"Train {train_index.shape[0]} samples.")
                print(f"Validation {validation_index.shape[0]} samples.")
                train_index_file = os.path.join(
                    self._results_path,
                    "folds",
                    f"fold_{fold_cnt}_train_indices.npy",
                )
                validation_index_file = os.path.join(
                    self._results_path,
                    "folds",
                    f"fold_{fold_cnt}_validation_indices.npy",
                )

                np.save(train_index_file, train_index)
                np.save(validation_index_file, validation_index)

        else:
            log.debug("Folds split already done, reuse it")

    def get_split(self, k, repeat=0):
        try:
            train_index_file = os.path.join(
                self._results_path, "folds", f"fold_{k}_train_indices.npy"
            )
            validation_index_file = os.path.join(
                self._results_path, "folds", f"fold_{k}_validation_indices.npy"
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

            train_data = {"X": X.iloc[train_index], "y": y.iloc[train_index]}
            validation_data = {
                "X": X.iloc[validation_index],
                "y": y.iloc[validation_index],
            }
            if sample_weight is not None:
                train_data["sample_weight"] = sample_weight.iloc[train_index]
                validation_data["sample_weight"] = sample_weight.iloc[validation_index]
        except Exception as e:
            import traceback

            print(traceback.format_exc())
            raise AutoMLException("Problem with custom validation. " + str(e))
        return (train_data, validation_data)

    def get_n_splits(self):
        return len(self.cv)

    def get_repeats(self):
        return 1
