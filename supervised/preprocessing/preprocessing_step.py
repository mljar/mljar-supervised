import copy
import pandas as pd
import numpy as np

from supervised.preprocessing.preprocessing_utils import PreprocessingUtils
from supervised.preprocessing.preprocessing_categorical import PreprocessingCategorical
from supervised.preprocessing.preprocessing_missing import PreprocessingMissingValues
from supervised.preprocessing.preprocessing_scale import PreprocessingScale
from supervised.preprocessing.label_encoder import LabelEncoder
from supervised.preprocessing.preprocessing_exclude_missing import (
    PreprocessingExcludeMissingValues,
)
import logging

log = logging.getLogger(__name__)


class PreprocessingStep(object):
    def __init__(self, preprocessing_params):
        self._params = preprocessing_params

        # preprocssing step attributes
        self._categorical_y

    def _exclude_missing_targets(self, X=None, y=None):
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

    def run(self, train_data, validation_data):
        log.info("PreprocessingStep.run")
        X_train, y_train = train_data.get("X"), train_data.get("y")
        X_validation, y_validation = validation_data.get("X"), validation_data.get("y")

        if y_train is not None:
            # target preprocessing
            # this must be used first, maybe we will drop some rows because of missing target values
            target_preprocessing = self._params.get("target_preprocessing")
            log.info("target_preprocessing -> {}".format(target_preprocessing))
            if PreprocessingMissingValues.NA_EXCLUDE in target_preprocessing:
                X_train, y_train = PreprocessingExcludeMissingValues.transform(X_train, y_train)
                X_validation, y_validation = PreprocessingExcludeMissingValues.transform(
                    X_validation, y_validation
                )

            if PreprocessingCategorical.CONVERT_INTEGER in target_preprocessing:
                self._categorical_y = LabelEncoder()
                self._categorical_y.fit(y_train)
                y_train = pd.Series(self._categorical_y.transform(y_train))
                if y_validation is not None and self._categorical_y is not None:
                    y_validation = pd.Series(self._categorical_y.transform(y_validation))

            if PreprocessingScale.SCALE_LOG_AND_NORMAL in target_preprocessing:
                log.error("not implemented SCALE_LOG_AND_NORMAL")
                raise Exception("not implemented SCALE_LOG_AND_NORMAL")

            if PreprocessingScale.SCALE_NORMAL in target_preprocessing:
                log.error("not implemented SCALE_NORMAL")
                raise Exception("not implemented SCALE_NORMAL")

        # columns preprocessing
        columns_preprocessing = self._params.get("columns_preprocessing")
        for preprocess_column in columns_preprocessing:
            log.info(
                "Preprocess column -> {}, {}".format(
                    preprocess_column, columns_preprocessing[preprocess_column]
                )
            )

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
                preprocessing_params["missing_values"] = mv
        if self._categorical is not None:
            cat = self._categorical.to_json()
            if cat:
                preprocessing_params["categorical"] = cat
        if self._categorical_y is not None:
            cat_y = self._categorical_y.to_json()
            if cat_y:
                preprocessing_params["categorical_y"] = cat_y
        return preprocessing_params

    def from_json(self, data_json):
        if "missing_values" in data_json:
            self._missing_values = PreprocessingMissingValues()
            self._missing_values.from_json(data_json["missing_values"])
        if "categorical" in data_json:
            self._categorical = PreprocessingCategorical()
            self._categorical.from_json(data_json["categorical"])
        if "categorical_y" in data_json:
            self._categorical_y = LabelEncoder()
            self._categorical_y.from_json(data_json["categorical_y"])
