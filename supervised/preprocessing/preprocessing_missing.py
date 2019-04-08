import os
import json
import numpy as np
import pandas as pd

from supervised.preprocessing.preprocessing_utils import PreprocessingUtils


class PreprocessingMissingValues(object):

    FILL_NA_MIN = "na_fill_min_1"
    FILL_NA_MEAN = "na_fill_mean"
    FILL_NA_MEDIAN = "na_fill_median"
    # there is no exlude in this class, because it requires working on both X and y!
    # Please check PreprocessingExcludeMissingValues
    NA_EXCLUDE = "na_exclude"
    MISSING_VALUE = "_missing_value_"
    REMOVE_COLUMN = "remove_column"

    def __init__(self, columns=[], na_fill_method=FILL_NA_MEDIAN):
        self._columns = columns
        # fill method
        self._na_fill_method = na_fill_method
        # fill parameters stored as a dict, feature -> fill value
        self._na_fill_params = {}

    def fit(self, X):
        X = self._fit_na_fill(X)

    def _fit_na_fill(self, X):
        for column in self._columns:
            if np.sum(pd.isnull(X[column]) == True) == 0:
                continue
            self._na_fill_params[column] = self._get_fill_value(X[column])

    def _get_fill_value(self, x):
        # categorical type
        if PreprocessingUtils.get_type(x) == PreprocessingUtils.CATEGORICAL:
            if self._na_fill_method == PreprocessingMissingValues.FILL_NA_MIN:
                return (
                    PreprocessingMissingValues.MISSING_VALUE
                )  # add new categorical value
            return PreprocessingUtils.get_most_frequent(x)
        # numerical type
        if self._na_fill_method == PreprocessingMissingValues.FILL_NA_MIN:
            return PreprocessingUtils.get_min(x) - 1.0
        if self._na_fill_method == PreprocessingMissingValues.FILL_NA_MEAN:
            return PreprocessingUtils.get_mean(x)
        return PreprocessingUtils.get_median(x)

    def transform(self, X):
        X = self._transform_na_fill(X)
        # this is additional run through columns,
        # in case of transforming data with new columns with missing values
        # X = self._make_sure_na_filled(X) # disbaled for now
        return X

    def _transform_na_fill(self, X):
        for column, value in self._na_fill_params.items():
            ind = pd.isnull(X.loc[:, column])
            X.loc[ind, column] = value
        return X

    def _make_sure_na_filled(self, X):
        self._fit_na_fill(X)
        return self._transform_na_fill(X)

    def to_json(self):
        # prepare json with all parameters
        if len(self._na_fill_params) == 0:
            return {}
        params = {
            "fill_method": self._na_fill_method,
            "fill_params": self._na_fill_params,
        }
        return params

    def from_json(self, params):
        if params is not None:
            self._na_fill_method = params.get("fill_method", None)
            self._na_fill_params = params.get("fill_params", {})
        else:
            self._na_fill_method, self._na_fill_params = None, None
