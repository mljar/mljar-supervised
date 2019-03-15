import os
import json
import numpy as np
import pandas as pd

from supervised.preprocessing.preprocessing_utils import PreprocessingUtils
from supervised.preprocessing.label_encoder import LabelEncoder
from supervised.preprocessing.label_binarizer import LabelBinarizer


class PreprocessingCategorical(object):

    CONVERT_ONE_HOT = "categorical_to_onehot"
    CONVERT_INTEGER = "categorical_to_int"

    def __init__(self, convert_categorical_method=CONVERT_INTEGER):
        self._convert_method = convert_categorical_method
        self._convert_params = {}

    def fit(self, x):
        self._fit_categorical_convert(x)

    def _fit_categorical_convert(self, x):
        if PreprocessingUtils.get_type(x) != PreprocessingUtils.CATEGORICAL:
            # no need to convert, already a number
            return
        # limit categories - it is needed when doing one hot encoding
        # this code is also used in predict.py file
        # and transform_utils.py
        # TODO it needs refactoring !!!
        too_much_categories = len(np.unique(list(x.values))) > 200
        lbl = None
        if (
            self._convert_method == PreprocessingCategorical.CONVERT_ONE_HOT
            and not too_much_categories
        ):
            lbl = LabelBinarizer()
            lbl.fit(x)
        else:
            lbl = LabelEncoder()
            lbl.fit(x)

        if lbl is not None:
            self._convert_params = lbl.to_json()

    def transform(self, x):

        lbl_params = self._convert_params
        if "unique_values" in lbl_params and "new_columns" in lbl_params:
            # convert to one hot
            lbl = LabelBinarizer()
            lbl.from_json(lbl_params)
            x = lbl.transform(x)
        else:
            # convert to integer
            lbl = LabelEncoder()
            lbl.from_json(lbl_params)
            x = lbl.transform(x)

        return x

    def to_json(self):
        if len(self._convert_params) == 0:
            return {}
        params = {
            "convert_method": self._convert_method,
            "convert_params": self._convert_params,
        }
        return params

    def from_json(self, params):
        if params is not None:
            self._convert_method = params.get("convert_method", None)
            self._convert_params = params.get("convert_params", {})
        else:
            self._convert_method, self._convert_params = None, None
