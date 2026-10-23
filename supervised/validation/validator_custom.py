import logging
import os

import joblib
import numpy as np

log = logging.getLogger(__name__)

from supervised.exceptions import AutoMLException
from supervised.validation.validator_base import BaseValidator


class CustomValidator(BaseValidator):

    def __init__(self, params, data_store=None):
        BaseValidator.__init__(self, params, data_store=data_store)

        cv_path = self.params.get("cv_path")

        if cv_path is None:
            raise AutoMLException("You need to specify `cv` as list or iterable")

        self.cv = joblib.load(cv_path)
        self.cv = list(self.cv)

        self._results_path = self.params.get("results_path")
        self._X_path = self.params.get("X_path")
        self._y_path = self.params.get("y_path")
        self._sample_weight_path = self.params.get("sample_weight_path")
        self._sensitive_features_path = self.params.get("sensitive_features_path")

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

            train_data = self._load_rows(train_index)
            validation_data = self._load_rows(validation_index)

        except Exception as e:
            import traceback

            print(traceback.format_exc())
            raise AutoMLException("Problem with custom validation. " + str(e))
        return (train_data, validation_data)

    def get_n_splits(self):
        return len(self.cv)

    def get_repeats(self):
        return 1
