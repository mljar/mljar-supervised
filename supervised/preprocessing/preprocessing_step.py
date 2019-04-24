import copy
import pandas as pd
import numpy as np
import warnings

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
    def __init__(
        self,
        preprocessing_params={"target_preprocessing": [], "columns_preprocessing": {}},
    ):
        self._params = preprocessing_params

        if "target_preprocessing" not in preprocessing_params:
            self._params["target_preprocessing"] = []
        if "columns_preprocessing" not in preprocessing_params:
            self._params["columns_preprocessing"] = {}

        # preprocssing step attributes
        self._categorical_y = None
        self._missing_values = []
        self._categorical = []
        self._scale = []
        self._remove_columns = []

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

    def run(self, train_data=None, validation_data=None):
        log.debug("PreprocessingStep.run")
        X_train, y_train = None, None
        if train_data is not None:
            if "X" in train_data:
                X_train = train_data.get("X").copy()
            if "y" in train_data:
                y_train = train_data.get("y").copy()
        X_validation, y_validation = None, None
        if validation_data is not None:
            if "X" in validation_data:
                X_validation = validation_data.get("X").copy()
            if "y" in validation_data:
                y_validation = validation_data.get("y").copy()

        if y_train is not None:
            # target preprocessing
            # this must be used first, maybe we will drop some rows because of missing target values
            target_preprocessing = self._params.get("target_preprocessing")
            log.debug("target_preprocessing -> {}".format(target_preprocessing))

            # if PreprocessingMissingValues.NA_EXCLUDE in target_preprocessing:
            X_train, y_train = PreprocessingExcludeMissingValues.transform(
                X_train, y_train
            )
            if validation_data is not None:
                X_validation, y_validation = PreprocessingExcludeMissingValues.transform(
                    X_validation, y_validation
                )

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
            log.debug("Preprocess column -> {}, {}".format(column, transforms))

        # remove empty or constant columns
        cols_to_remove = list(
            filter(
                lambda k: "remove_column" in columns_preprocessing[k],
                columns_preprocessing,
            )
        )

        if X_train is not None:
            X_train.drop(cols_to_remove, axis=1, inplace=True)
        if X_validation is not None:
            X_validation.drop(cols_to_remove, axis=1, inplace=True)
        self._remove_columns = cols_to_remove

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

        # SCALE
        for scale_method in [PreprocessingScale.SCALE_NORMAL]:
            cols_to_process = list(
                filter(
                    lambda k: scale_method in columns_preprocessing[k],
                    columns_preprocessing,
                )
            )
            if len(cols_to_process):
                scale = PreprocessingScale(cols_to_process)
                scale.fit(X_train)
                X_train = scale.transform(X_train)
                if X_validation is not None:
                    X_validation = scale.transform(X_validation)
                self._scale += [scale]

        return {"X": X_train, "y": y_train}, {"X": X_validation, "y": y_validation}

    def transform(self, validation_data=None):
        log.debug("PreprocessingStep.transform")
        X_validation, y_validation = None, None
        if validation_data is not None:
            if "X" in validation_data:
                X_validation = validation_data.get("X").copy()
            if "y" in validation_data:
                y_validation = validation_data.get("y").copy()

        # target preprocessing
        # this must be used first, maybe we will drop some rows because of missing target values
        target_preprocessing = self._params.get("target_preprocessing")
        log.debug("target_preprocessing -> {}".format(target_preprocessing))

        if validation_data is not None:
            X_validation, y_validation = PreprocessingExcludeMissingValues.transform(
                X_validation, y_validation
            )

        if PreprocessingCategorical.CONVERT_INTEGER in target_preprocessing:
            if y_validation is not None and self._categorical_y is not None:
                y_validation = pd.Series(self._categorical_y.transform(y_validation))

        if PreprocessingScale.SCALE_LOG_AND_NORMAL in target_preprocessing:
            log.error("not implemented SCALE_LOG_AND_NORMAL")
            raise Exception("not implemented SCALE_LOG_AND_NORMAL")

        if PreprocessingScale.SCALE_NORMAL in target_preprocessing:
            log.error("not implemented SCALE_NORMAL")
            raise Exception("not implemented SCALE_NORMAL")

        # columns preprocessing
        if len(self._remove_columns) and X_validation is not None:
            cols_to_remove = [
                col for col in X_validation.columns if col in self._remove_columns
            ]
            X_validation.drop(cols_to_remove, axis=1, inplace=True)

        for missing in self._missing_values:
            if X_validation is not None and missing is not None:
                X_validation = missing.transform(X_validation)
        # to be sure that all missing are filled
        # in case new data there can be gaps!
        if np.sum(np.sum(pd.isnull(X_validation))) > 0:
            # there is something missing, fill it
            # we should notice user about it!
            warnings.warn(
                "There are columns {} with missing values which didnt have missing values in train dataset.".format(
                    list(X_validation.columns[np.where(np.sum(pd.isnull(X_validation)))])
                )
            )
            missing = PreprocessingMissingValues(
                X_validation.columns, PreprocessingMissingValues.FILL_NA_MEDIAN
            )
            missing.fit(X_validation)
            X_validation = missing.transform(X_validation)
        for convert in self._categorical:
            if X_validation is not None and convert is not None:
                X_validation = convert.transform(X_validation)
        for scale in self._scale:
            if X_validation is not None and scale is not None:
                X_validation = scale.transform(X_validation)

        return {"X": X_validation, "y": y_validation}

    def to_json(self):
        preprocessing_params = {}
        if self._remove_columns:
            preprocessing_params["remove_columns"] = self._remove_columns
        if self._missing_values is not None and len(self._missing_values):
            mvs = []  # refactor
            for mv in self._missing_values:
                if mv.to_json():
                    mvs += [mv.to_json()]
            if mvs:
                preprocessing_params["missing_values"] = mvs
        if self._categorical is not None and len(self._categorical):
            cats = []  # refactor
            for cat in self._categorical:
                if cat.to_json():
                    cats += [cat.to_json()]
            if cats:
                preprocessing_params["categorical"] = cats
        if self._scale is not None and len(self._scale):
            scs = [sc.to_json() for sc in self._scale if sc.to_json()]
            if scs:
                preprocessing_params["scale"] = scs
        if self._categorical_y is not None:
            cat_y = self._categorical_y.to_json()
            if cat_y:
                preprocessing_params["categorical_y"] = cat_y
        return preprocessing_params

    def reverse_transform_target(self, y):

        # target_preprocessing = self._params.get("target_preprocessing")
        # assume for now that all tasks are binary classification
        # if there is no target preprocessing, assume that there is 0 and 1 target
        pos_label, neg_label = "1", "0"
        if self._categorical_y is not None:
            for label, value in self._categorical_y.to_json().items():
                if value == 1:
                    pos_label = label
                else:
                    neg_label = label

        return pd.DataFrame(
            {"p_{}".format(neg_label): 1 - y, "p_{}".format(pos_label): y}
        )

    def from_json(self, data_json):
        if "remove_columns" in data_json:
            self._remove_columns = data_json.get("remove_columns", [])
        if "missing_values" in data_json:
            self._missing_values = []
            for mv_data in data_json["missing_values"]:
                mv = PreprocessingMissingValues()
                mv.from_json(mv_data)
                self._missing_values += [mv]
        if "categorical" in data_json:
            self._categorical = []
            for cat_data in data_json["categorical"]:
                cat = PreprocessingCategorical()
                cat.from_json(cat_data)
                self._categorical += [cat]
        if "scale" in data_json:
            self._scale = []
            for scale_data in data_json["scale"]:
                sc = PreprocessingScale()
                sc.from_json(scale_data)
                self._scale += [sc]
        if "categorical_y" in data_json:
            self._categorical_y = LabelEncoder()
            self._categorical_y.from_json(data_json["categorical_y"])
