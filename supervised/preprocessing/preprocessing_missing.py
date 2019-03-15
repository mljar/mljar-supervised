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

    def __init__(self, na_fill_method=FILL_NA_MEDIAN):
        # fill method
        self._na_fill_method = na_fill_method
        # fill parameters stored as a dict, feature -> fill value
        self._na_fill_params = None

    # for one column
    def fit(self, x):
        x = self._fit_na_fill(x)

    def _fit_na_fill(self, x):
        if np.sum(pd.isnull(x) == True) == 0:
            return
        self._na_fill_params = self._get_fill_value(x)

    def transform(self, x):
        x = self._transform_na_fill(x)
        # this is additional run through columns,
        # in case of transforming data with new columns with missing values
        x = self._make_sure_na_filled(x)
        return x

    def _transform_na_fill(self, x):
        value = self._na_fill_params
        ind = pd.isnull(x)
        print(x.is_copy, "copy")
        x.loc[ind] = value ###########################x.loc[ind]
        return x

    def to_json(self):
        # prepare json with all parameters
        if self._na_fill_params is None:
            return {}
        params = {
            "fill_method": self._na_fill_method,
            "fill_params": self._na_fill_params,
        }
        return params

    def from_json(self, params):
        if params is not None:
            self._na_fill_method = params.get("fill_method", None)
            self._na_fill_params = params.get("fill_params", None)
        else:
            self._na_fill_method, self._na_fill_params = None, None

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

    def _make_sure_na_filled(self, x):
        self._fit_na_fill(x)
        return self._transform_na_fill(x)
