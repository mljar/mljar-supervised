import os
import json
import numpy as np
import pandas as pd

import logging

log = logging.getLogger(__name__)


class PreprocessingExcludeMissingValues(object):
    @staticmethod
    def transform(self, X=None, y=None):
        log.info("Exclude rows with missing target values")
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
