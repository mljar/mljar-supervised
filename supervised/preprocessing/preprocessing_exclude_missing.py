import os
import json
import numpy as np
import pandas as pd

import logging

log = logging.getLogger(__name__)


class PreprocessingExcludeMissingValues(object):
    @staticmethod
    def remove_rows_without_target(data):
        if "train" in data:
            X_train = data.get("train").get("X")
            y_train = data.get("train").get("y")
            X_train, y_train = PreprocessingExcludeMissingValues.transform(
                X_train, y_train
            )
            data["train"]["X"] = X_train
            data["train"]["y"] = y_train
        if "validation" in data:
            X_validation = data.get("validation").get("X")
            y_validation = data.get("validation").get("y")
            X_validation, y_validation = PreprocessingExcludeMissingValues.transform(
                X_validation, y_validation
            )
            data["validation"]["X"] = X_validation
            data["validation"]["y"] = y_validation
        return data

    @staticmethod
    def transform(X=None, y=None):
        log.debug("Exclude rows with missing target values")
        if y is None:
            return X, y
        y_missing = pd.isnull(y)
        if np.sum(np.array(y_missing)) == 0:
            return X, y
        y = y.drop(y.index[y_missing])
        y.index = range(y.shape[0])
        if X is not None:
            X = X.drop(X.index[y_missing])
            X.index = range(X.shape[0])
        return X, y
