import logging
import numpy as np

log = logging.getLogger(__name__)

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold

from supervised.validation.validator_base import BaseValidator


class KFoldValidator(BaseValidator):
    def __init__(self, params, data):
        BaseValidator.__init__(self, params, data)

        self.k_folds = self.params.get("k_folds", 5)
        self.shuffle = self.params.get("shuffle", True)
        self.stratify = self.params.get("stratify", False)
        self.random_seed = self.params.get("random_seed", 1706)

        if self.stratify:
            self.skf = StratifiedKFold(
                n_splits=self.k_folds,
                shuffle=self.shuffle,
                random_state=self.random_seed,
            )
        else:
            self.skf = KFold(
                n_splits=self.k_folds,
                shuffle=self.shuffle,
                random_state=self.random_seed,
            )

    def split(self):
        X = self.data["train"]["X"]
        y = self.data["train"]["y"]

        for train_index, validation_index in self.skf.split(X, y):
            X_train = X.loc[train_index]
            y_train = y.loc[train_index]
            X_validation = X.loc[validation_index]
            y_validation = y.loc[validation_index]
            yield {"X": X_train, "y": y_train}, {"X": X_validation, "y": y_validation}

    def get_n_splits(self):
        return self.k_folds
