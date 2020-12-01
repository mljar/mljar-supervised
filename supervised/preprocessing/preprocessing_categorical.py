import os
import json
import numpy as np
import pandas as pd

from supervised.preprocessing.preprocessing_utils import PreprocessingUtils
from supervised.preprocessing.label_encoder import LabelEncoder
from supervised.preprocessing.label_binarizer import LabelBinarizer
from supervised.preprocessing.loo_encoder import LooEncoder


class PreprocessingCategorical(object):

    CONVERT_ONE_HOT = "categorical_to_onehot"
    CONVERT_INTEGER = "categorical_to_int"
    CONVERT_LOO = "categorical_to_loo"

    FEW_CATEGORIES = "few_categories"
    MANY_CATEGORIES = "many_categories"

    def __init__(self, columns=[], method=CONVERT_INTEGER):
        self._convert_method = method
        self._convert_params = {}
        self._columns = columns
        self._enc = None

    def fit(self, X, y=None):
        if (
            self._convert_method == PreprocessingCategorical.CONVERT_LOO
            and self._columns
        ):
            self._enc = LooEncoder(cols=self._columns)
            self._enc.fit(X, y)
        else:
            self._fit_categorical_convert(X)

    def _fit_categorical_convert(self, X):
        for column in self._columns:

            if PreprocessingUtils.get_type(X[column]) != PreprocessingUtils.CATEGORICAL:
                # no need to convert, already a number
                continue
            # limit categories - it is needed when doing one hot encoding
            # this code is also used in predict.py file
            # and transform_utils.py
            # TODO it needs refactoring !!!
            too_much_categories = len(np.unique(list(X[column].values))) > 200
            lbl = None
            if (
                self._convert_method == PreprocessingCategorical.CONVERT_ONE_HOT
                and not too_much_categories
            ):
                lbl = LabelBinarizer()
                lbl.fit(X, column)
            else:
                lbl = LabelEncoder()
                lbl.fit(X[column])

            if lbl is not None:
                self._convert_params[column] = lbl.to_json()

    def transform(self, X):
        if (
            self._convert_method == PreprocessingCategorical.CONVERT_LOO
            and self._columns
        ):
            return self._enc.transform(X)
        else:
            for column, lbl_params in self._convert_params.items():
                if "unique_values" in lbl_params and "new_columns" in lbl_params:
                    # convert to one hot
                    lbl = LabelBinarizer()
                    lbl.from_json(lbl_params)
                    X = lbl.transform(X, column)
                else:
                    # convert to integer
                    lbl = LabelEncoder()
                    lbl.from_json(lbl_params)
                    X.loc[:, column] = lbl.transform(X.loc[:, column])

            return X

    def inverse_transform(self, X):
        for column, lbl_params in self._convert_params.items():
            if "unique_values" in lbl_params and "new_columns" in lbl_params:
                # convert to one hot
                lbl = LabelBinarizer()
                lbl.from_json(lbl_params)
                X = lbl.inverse_transform(X, column)  # should raise exception
            else:
                # convert to integer
                lbl = LabelEncoder()
                lbl.from_json(lbl_params)
                X.loc[:, column] = lbl.inverse_transform(X.loc[:, column])

        return X

    def to_json(self):
        params = {}
        if (
            self._convert_method == PreprocessingCategorical.CONVERT_LOO
            and self._columns
        ):
            params = {
                "enc": self._enc.to_json(),
                "convert_method": self._convert_method,
                "columns": self._columns,
            }
        else:
            if len(self._convert_params) == 0:
                return {}
            params = {
                "convert_method": self._convert_method,
                "convert_params": self._convert_params,
                "columns": self._columns,
            }
        return params

    def from_json(self, params):
        if params is not None:
            self._convert_method = params.get("convert_method", None)
            self._columns = params.get("columns", [])
            if self._convert_method == PreprocessingCategorical.CONVERT_LOO:
                self._enc = LooEncoder()
                self._enc.from_json(params.get("enc", {}))
            else:
                self._convert_params = params.get("convert_params", {})

        else:
            self._convert_method, self._convert_params = None, None
            self._columns = []
