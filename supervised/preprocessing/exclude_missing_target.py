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
    def transform(X=None, y=None, sample_weight=None, warn=False):
        if y is None:
            return X, y, sample_weight
        y_missing = pd.isnull(y)
        if np.sum(np.array(y_missing)) == 0:
            return X, y, sample_weight
        logger.debug("Exclude rows with missing target values")
        if warn:
            warnings.warn(
                "There are samples with missing target values in the data which will be excluded for further analysis"
            )
        y = y.drop(y.index[y_missing])
        y.reset_index(drop=True, inplace=True)

        if X is not None:
            X = X.drop(X.index[y_missing])
            X.reset_index(drop=True, inplace=True)

        if sample_weight is not None:
            sample_weight = sample_weight.drop(sample_weight.index[y_missing])
            sample_weight.reset_index(drop=True, inplace=True)

        return X, y, sample_weight
