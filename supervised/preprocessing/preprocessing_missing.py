import os
import json
import numpy as np
import pandas as pd

from supervised.preprocessing.preprocessing_utils import PreprocessingUtils


class PreprocessingMissingValues(object):

    FILL_NA_MIN = "na_fill_min_1"
    FILL_NA_MEAN = "na_fill_mean"
    FILL_NA_MEDIAN = "na_fill_median"
    FILL_DATETIME = "na_fill_datetime"

    NA_EXCLUDE = "na_exclude"
    MISSING_VALUE = "_missing_value_"
    REMOVE_COLUMN = "remove_column"

    def __init__(self, columns=[], na_fill_method=FILL_NA_MEDIAN):
        self._columns = columns
        # fill method
        self._na_fill_method = na_fill_method
        # fill parameters stored as a dict, feature -> fill value
        self._na_fill_params = {}
        self._datetime_columns = []

    def fit(self, X):
        X = self._fit_na_fill(X)

    def _fit_na_fill(self, X):
        for column in self._columns:
            if np.sum(pd.isnull(X[column]) == True) == 0:
                continue
            self._na_fill_params[column] = self._get_fill_value(X[column])
            if PreprocessingUtils.get_type(X[column]) == PreprocessingUtils.DATETIME:
                self._datetime_columns += [column]

    def _get_fill_value(self, x):
        # categorical type
        if PreprocessingUtils.get_type(x) == PreprocessingUtils.CATEGORICAL:
            if self._na_fill_method == PreprocessingMissingValues.FILL_NA_MIN:
                return (
                    PreprocessingMissingValues.MISSING_VALUE
                )  # add new categorical value
            return PreprocessingUtils.get_most_frequent(x)
        # datetime
        if PreprocessingUtils.get_type(x) == PreprocessingUtils.DATETIME:
            return PreprocessingUtils.get_most_frequent(x)
        # text
        if PreprocessingUtils.get_type(x) == PreprocessingUtils.TEXT:
            return PreprocessingMissingValues.MISSING_VALUE

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
            "datetime_columns": list(self._datetime_columns),
        }
        for col in self._datetime_columns:
            params["fill_params"][col] = str(params["fill_params"][col])
        return params

    def from_json(self, params):
        if params is not None:
            self._na_fill_method = params.get("fill_method", None)
            self._na_fill_params = params.get("fill_params", {})
            self._datetime_columns = params.get("datetime_columns", [])
            for col in self._datetime_columns:
                self._na_fill_params[col] = pd.to_datetime(self._na_fill_params[col])
        else:
            self._na_fill_method, self._na_fill_params = None, None
            self._datetime_columns = []
