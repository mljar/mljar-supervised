import os
import json
import warnings
import numpy as np
import pandas as pd

import logging
from supervised.utils.config import LOG_LEVEL

logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)


class ExcludeRowsMissingTarget(object):
    @staticmethod
    def remove_rows_without_target(data):
        if "train" in data:
            X_train = data.get("train").get("X")
            y_train = data.get("train").get("y")
            X_train, y_train = ExcludeRowsMissingTarget.transform(X_train, y_train)
            data["train"]["X"] = X_train
            data["train"]["y"] = y_train
        if "validation" in data:
            X_validation = data.get("validation").get("X")
            y_validation = data.get("validation").get("y")
            X_validation, y_validation = ExcludeRowsMissingTarget.transform(
                X_validation, y_validation
            )
            data["validation"]["X"] = X_validation
            data["validation"]["y"] = y_validation
        return data

    @staticmethod
    def transform(X=None, y=None, warn=False):
        if y is None:
            return X, y
        y_missing = pd.isnull(y)
        if np.sum(np.array(y_missing)) == 0:
            return X, y
        logger.debug("Exclude rows with missing target values")
        if warn:
            warnings.warn(
                "There are samples with missing target values in the data which will be excluded for further analysis"
            )
        y = y.drop(y.index[y_missing])
        y.reset_index(drop=True, inplace=True)
        # y.index = range(y.shape[0])
        if X is not None:
            X = X.drop(X.index[y_missing])
            X.reset_index(drop=True, inplace=True)
            # X.index = range(X.shape[0])
        return X, y
