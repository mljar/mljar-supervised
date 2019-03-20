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
    def __init__(self, preprocessing_params={"target_preprocessing": [], "columns_preprocessing": {}}):
        self._params = preprocessing_params

        # preprocssing step attributes
        self._categorical_y = None
        self._missing_values = []
        self._categorical = []

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

    def run(self, train_data, validation_data=None):
        log.info("PreprocessingStep.run")
        X_train, y_train = train_data.get("X"), train_data.get("y")
        X_validation, y_validation = None, None
        if validation_data is not None:
            X_validation, y_validation = validation_data.get("X"), validation_data.get("y")

        if y_train is not None:
            # target preprocessing
            # this must be used first, maybe we will drop some rows because of missing target values
            target_preprocessing = self._params.get("target_preprocessing")
            log.info("target_preprocessing -> {}".format(target_preprocessing))
            print(X_train.shape)
            print(y_train.shape)
            #if PreprocessingMissingValues.NA_EXCLUDE in target_preprocessing:
            X_train, y_train = PreprocessingExcludeMissingValues.transform(
                X_train, y_train
            )
            if validation_data is not None:
                X_validation, y_validation = PreprocessingExcludeMissingValues.transform(
                    X_validation, y_validation
                )
            print(X_train.shape)
            print(y_train.shape)
            if PreprocessingCategorical.CONVERT_INTEGER in target_preprocessing:
                self._categorical_y = LabelEncoder()
                self._categorical_y.fit(y_train)
                y_train = pd.Series(self._categorical_y.transform(y_train))
                if y_validation is not None and self._categorical_y is not None:
                    y_validation = pd.Series(
                        self._categorical_y.transform(y_validation)
                    )

            if PreprocessingScale.SCALE_LOG_AND_NORMAL in target_preprocessing:
                log.error("not implemented SCALE_LOG_AND_NORMAL")
                raise Exception("not implemented SCALE_LOG_AND_NORMAL")

            if PreprocessingScale.SCALE_NORMAL in target_preprocessing:
                log.error("not implemented SCALE_NORMAL")
                raise Exception("not implemented SCALE_NORMAL")

        # columns preprocessing
        columns_preprocessing = self._params.get("columns_preprocessing")
        for column in columns_preprocessing:
            transforms = columns_preprocessing[column]
            log.info("Preprocess column -> {}, {}".format(column, transforms))

        for missing_method in [PreprocessingMissingValues.FILL_NA_MEDIAN]:
            cols_to_process = list(
                filter(
                    lambda k: missing_method in columns_preprocessing[k],
                    columns_preprocessing,
                )
            )
            missing = PreprocessingMissingValues(cols_to_process, missing_method)
            missing.fit(X_train)
            X_train = missing.transform(X_train)
            if X_validation is not None:
                X_validation = missing.transform(X_validation)
            self._missing_values += [missing]

        for convert_method in [PreprocessingCategorical.CONVERT_INTEGER]:
            cols_to_process = list(
                filter(
                    lambda k: convert_method in columns_preprocessing[k],
                    columns_preprocessing,
                )
            )
            convert = PreprocessingCategorical(cols_to_process, convert_method)
            convert.fit(X_train)
            X_train = convert.transform(X_train)
            if X_validation is not None:
                X_validation = convert.transform(X_validation)
            self._categorical += [convert]

        return {"X": X_train, "y": y_train}, {"X": X_validation, "y": y_validation}

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
