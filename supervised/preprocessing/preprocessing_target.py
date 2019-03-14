import copy
import pandas as pd
import numpy as np

from supervised.preprocessing.preprocessi_utils import PreprocessingUtils
from supervised.preprocessing.preprocessing_categorical import PreprocessingCategorical
from supervised.preprocessing.preprocessing_missing import PreprocessingMissingValues
from supervised.preprocessing.preprocessing_scale import PreprocessingScale


class PreprocessingTarget(object):
    def __init__(self):
        pass

    def run(self, X_train, y_train=None, X_test=None, y_test=None):
        pass

    def fit(self, X):
        # remove missing values
        pass
        # categorical convert
        pass

    def transform(self, X):
        # missing values
        if self._missing_values is not None:
            X = self._missing_values.transform(X)
        # catagorical
        if self._categorical is not None:
            X = self._categorical.transform(X)
        return X

    def to_json(self):
        pass

    def from_json(self, data_json):
        pass
