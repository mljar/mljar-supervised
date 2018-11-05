import copy
import pandas as pd
import numpy as np

from utils.jsonable import Jsonable
from preprocessing_utils import PreprocessingUtils
from preprocessing_categorical import PreprocessingCategorical
from preprocessing_missing import PreprocessingMissingValues
from preprocessing_scale import PreprocessingScale
from preprocessing_box import PreprocessingBox
from preprocessing.label_encoder import LabelEncoder

class PreprocessingStep(Jsonable):

    def __init__(self, missing_values_method = PreprocessingMissingValues.FILL_NA_MEDIAN, \
                        categorical_method = PreprocessingCategorical.CONVERT_INTEGER,
                        scale_method = PreprocessingScale.SCALE_NORMAL,
                        project_task = 'PROJECT_BIN_CLASS'):
        self._missing_values_method = missing_values_method
        self._categorical_method = categorical_method
        self._scale_method = scale_method
        self._project_task = project_task

        self._missing_values = None if self._missing_values_method is None else PreprocessingMissingValues(self._missing_values_method)
        self._categorical = None if self._categorical_method is None else PreprocessingCategorical(self._categorical_method)
        self._scale = None
        self._categorical_y = None

    def _exclude_missing_targets(self, X = None, y = None):
        # check if there are missing values in target column
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

    def run(self, X_train = None, y_train = None, X_test = None, y_test = None):
        # check if there are missing values

        X_train, y_train = self._exclude_missing_targets(X_train, y_train)
        X_test, y_test = self._exclude_missing_targets(X_test, y_test)

        if y_train is not None:
            apply_convert = False
            if self._project_task == 'PROJECT_BIN_CLASS':
                u = np.unique(y_train)
                apply_convert = not(0 in u and 1 in u)

            if apply_convert or PreprocessingUtils.CATEGORICAL == PreprocessingUtils.get_type(y_train):
                self._categorical_y = LabelEncoder()
                self._categorical_y.fit(y_train)
                y_train = pd.Series(self._categorical_y.transform(y_train))

        if y_test is not None and self._categorical_y is not None:
            y_test = pd.Series(self._categorical_y.transform(y_test))
        if X_train is not None:
            # missing values
            if self._missing_values is not None:
                self._missing_values.fit(X_train)
                X_train = self._missing_values.transform(X_train)
            # categorical
            if self._categorical is not None:
                self._categorical.fit(X_train)
                X_train = self._categorical.transform(X_train)

        # apply missing and categorical transforms
        if X_test is not None:
            if self._missing_values is not None:
                X_test = self._missing_values.transform(X_test)
            if self._categorical is not None:
                X_test = self._categorical.transform(X_test)

        return X_train, y_train, X_test, y_test


    def to_json(self):
        preprocessing_params = {}
        if self._missing_values is not None:
            mv = self._missing_values.to_json()
            if mv:
                preprocessing_params['missing_values'] = mv
        if self._categorical is not None:
            cat = self._categorical.to_json()
            if cat:
                preprocessing_params['categorical'] = cat
        if self._categorical_y is not None:
            cat_y = self._categorical_y.to_json()
            if cat_y:
                preprocessing_params['categorical_y'] = cat_y
        return preprocessing_params

    def from_json(self, data_json):
        if 'missing_values' in data_json:
            self._missing_values = PreprocessingMissingValues()
            self._missing_values.from_json(data_json['missing_values'])
        if 'categorical' in data_json:
            self._categorical = PreprocessingCategorical()
            self._categorical.from_json(data_json['categorical'])
        if 'categorical_y' in data_json:
            self._categorical_y = LabelEncoder()
            self._categorical_y.from_json(data_json['categorical_y'])
