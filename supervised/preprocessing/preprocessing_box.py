import copy
import pandas as pd
import numpy as np

from utils.jsonable import Jsonable
from preprocessing_utils import PreprocessingUtils
from preprocessing_categorical import PreprocessingCategorical
from preprocessing_missing import PreprocessingMissingValues
from preprocessing_scale import PreprocessingScale

class PreprocessingBox(Jsonable):

    def __init__(self, missing_values_method = PreprocessingMissingValues.FILL_NA_MEDIAN, \
                        categorical_method = PreprocessingCategorical.CONVERT_INTEGER,
                        scale_method = PreprocessingScale.SCALE_NORMAL):
        self._missing_values_method = missing_values_method
        self._categorical_method = categorical_method
        self._scale_method = scale_method

        self._missing_values = None
        self._categorical = None
        self._scale = None

    def run(self, X_train, y_train = None, X_test = None, y_test = None):
        pass

    def fit(self, X):
        # missing values
        if self._missing_values_method is not None:
            self._missing_values = PreprocessingMissingValues(self._missing_values_method)
            self._missing_values.fit(X)

        # categorical convert
        if self._categorical_method is not None:

            self._categorical = PreprocessingCategorical(self._categorical_method)
            if self._missing_values is not None:
                X_copy = copy.deepcopy(X)
                X_copy = self._missing_values.transform(X_copy)
                self._categorical.fit(X_copy)
                del X_copy
            else:
                self._categorical.fit(X)

    def transform(self, X):
        # missing values
        if self._missing_values is not None:
            X = self._missing_values.transform(X)
        # catagorical
        if self._categorical is not None:
            X = self._categorical.transform(X)
        return X

    def to_json(self):
        data_json = {}
        if self._missing_values is not None:
            mvj = self._missing_values.to_json()
            if mvj:
                data_json['missing_values'] = mvj
        if self._categorical is not None:
            cj = self._categorical.to_json()
            if cj:
                data_json['categorical'] = cj
        return data_json

    def from_json(self, data_json):
        if 'missing_values' in data_json:
            self._missing_values_method = data_json['missing_values']['fill_method']
            self._missing_values = PreprocessingMissingValues(self._missing_values_method)
            self._missing_values.from_json(data_json['missing_values'])
        if 'categorical' in data_json:
            self._categorical_method = data_json['categorical']['convert_method']
            self._categorical = PreprocessingCategorical(self._categorical_method)
            self._categorical.from_json(data_json['categorical'])
