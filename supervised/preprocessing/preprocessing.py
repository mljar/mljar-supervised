import copy
import pandas as pd
import numpy as np
import warnings
import logging

from supervised.preprocessing.preprocessing_utils import PreprocessingUtils
from supervised.preprocessing.preprocessing_categorical import PreprocessingCategorical
from supervised.preprocessing.preprocessing_missing import PreprocessingMissingValues
from supervised.preprocessing.scale import Scale
from supervised.preprocessing.label_encoder import LabelEncoder
from supervised.preprocessing.label_binarizer import LabelBinarizer
from supervised.preprocessing.datetime_transformer import DateTimeTransformer
from supervised.preprocessing.text_transformer import TextTransformer
from supervised.preprocessing.goldenfeatures_transformer import (
    GoldenFeaturesTransformer,
)
from supervised.preprocessing.kmeans_transformer import KMeansTransformer

from supervised.preprocessing.exclude_missing_target import ExcludeRowsMissingTarget
from supervised.algorithms.registry import (
    BINARY_CLASSIFICATION,
    MULTICLASS_CLASSIFICATION,
    REGRESSION,
)
from supervised.utils.config import LOG_LEVEL
from supervised.exceptions import AutoMLException

logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)


class Preprocessing(object):
    def __init__(
        self,
        preprocessing_params={"target_preprocessing": [], "columns_preprocessing": {}},
        model_name =None,
        k_fold = None, repeat = None
    ):
        self._params = preprocessing_params

        if "target_preprocessing" not in preprocessing_params:
            self._params["target_preprocessing"] = []
        if "columns_preprocessing" not in preprocessing_params:
            self._params["columns_preprocessing"] = {}

        # preprocssing step attributes
        self._categorical_y = None
        self._scale_y = None
        self._missing_values = []
        self._categorical = []
        self._scale = []
        self._remove_columns = []
        self._datetime_transforms = []
        self._text_transforms = []
        self._golden_features = None
        self._kmeans = None
        self._add_random_feature = self._params.get("add_random_feature", False)
        self._drop_features = self._params.get("drop_features", [])
        self._model_name = model_name
        self._k_fold = k_fold
        self._repeat = repeat

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

    # fit and transform
    def fit_and_transform(self, X_train, y_train, sample_weight=None):
        logger.debug("Preprocessing.fit_and_transform")

        if y_train is not None:
            # target preprocessing
            # this must be used first, maybe we will drop some rows because of missing target values
            target_preprocessing = self._params.get("target_preprocessing")
            logger.debug("target_preprocessing params: {}".format(target_preprocessing))

            X_train, y_train, sample_weight = ExcludeRowsMissingTarget.transform(
                X_train, y_train, sample_weight
            )

            if PreprocessingCategorical.CONVERT_INTEGER in target_preprocessing:
                logger.debug("Convert target to integer")
                self._categorical_y = LabelEncoder(try_to_fit_numeric=True)
                self._categorical_y.fit(y_train)
                y_train = pd.Series(self._categorical_y.transform(y_train))

            if PreprocessingCategorical.CONVERT_ONE_HOT in target_preprocessing:
                logger.debug("Convert target to one-hot coding")
                self._categorical_y = LabelBinarizer()
                self._categorical_y.fit(pd.DataFrame({"target": y_train}), "target")
                y_train = self._categorical_y.transform(
                    pd.DataFrame({"target": y_train}), "target"
                )

            if Scale.SCALE_LOG_AND_NORMAL in target_preprocessing:
                logger.debug("Scale log and normal")

                self._scale_y = Scale(
                    ["target"], scale_method=Scale.SCALE_LOG_AND_NORMAL
                )
                y_train = pd.DataFrame({"target": y_train})
                self._scale_y.fit(y_train)
                y_train = self._scale_y.transform(y_train)
                y_train = y_train["target"]

            if Scale.SCALE_NORMAL in target_preprocessing:
                logger.debug("Scale normal")

                self._scale_y = Scale(["target"], scale_method=Scale.SCALE_NORMAL)
                y_train = pd.DataFrame({"target": y_train})
                self._scale_y.fit(y_train)
                y_train = self._scale_y.transform(y_train)
                y_train = y_train["target"]

        # columns preprocessing
        columns_preprocessing = self._params.get("columns_preprocessing")
        for column in columns_preprocessing:
            transforms = columns_preprocessing[column]
            # logger.debug("Preprocess column {} with: {}".format(column, transforms))

        # remove empty or constant columns
        cols_to_remove = list(
            filter(
                lambda k: "remove_column" in columns_preprocessing[k],
                columns_preprocessing,
            )
        )

        if X_train is not None:
            X_train.drop(cols_to_remove, axis=1, inplace=True)
        self._remove_columns = cols_to_remove

        numeric_cols = []  # get numeric cols before text transformations
        # needed for golden features
        if X_train is not None and (
            "golden_features" in self._params or "kmeans_features" in self._params
        ):
            numeric_cols = X_train.select_dtypes(include="number").columns.tolist()

        # there can be missing values in the text data,
        # but we don't want to handle it by fill missing methods
        # zeros will be imputed by text_transform method
        cols_to_process = list(
            filter(
                lambda k: "text_transform" in columns_preprocessing[k],
                columns_preprocessing,
            )
        )

        new_text_columns = []
        for col in cols_to_process:
            t = TextTransformer()
            t.fit(X_train, col)
            X_train = t.transform(X_train)
            self._text_transforms += [t]
            new_text_columns += t._new_columns
        # end of text transform

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
            self._missing_values += [missing]

        # golden features
        golden_columns = []
        if "golden_features" in self._params:
            results_path = self._params["golden_features"]["results_path"]
            ml_task = self._params["golden_features"]["ml_task"]
            self._golden_features = GoldenFeaturesTransformer(results_path, ml_task)
            self._golden_features.fit(X_train[numeric_cols], y_train)
            X_train = self._golden_features.transform(X_train)
            golden_columns = self._golden_features._new_columns

        kmeans_columns = []
        if "kmeans_features" in self._params:
            results_path = self._params["kmeans_features"]["results_path"]
            self._kmeans = KMeansTransformer(results_path, self._model_name, self._k_fold)
            self._kmeans.fit(X_train[numeric_cols], y_train)
            X_train = self._kmeans.transform(X_train)
            kmeans_columns = self._kmeans._new_features

        for convert_method in [
            PreprocessingCategorical.CONVERT_INTEGER,
            PreprocessingCategorical.CONVERT_ONE_HOT,
            PreprocessingCategorical.CONVERT_LOO,
        ]:
            cols_to_process = list(
                filter(
                    lambda k: convert_method in columns_preprocessing[k],
                    columns_preprocessing,
                )
            )
            convert = PreprocessingCategorical(cols_to_process, convert_method)
            convert.fit(X_train, y_train)
            X_train = convert.transform(X_train)
            self._categorical += [convert]

        # datetime transform
        cols_to_process = list(
            filter(
                lambda k: "datetime_transform" in columns_preprocessing[k],
                columns_preprocessing,
            )
        )

        new_datetime_columns = []
        for col in cols_to_process:

            t = DateTimeTransformer()
            t.fit(X_train, col)
            X_train = t.transform(X_train)
            self._datetime_transforms += [t]
            new_datetime_columns += t._new_columns

        # SCALE
        for scale_method in [Scale.SCALE_NORMAL, Scale.SCALE_LOG_AND_NORMAL]:
            cols_to_process = list(
                filter(
                    lambda k: scale_method in columns_preprocessing[k],
                    columns_preprocessing,
                )
            )
            if (
                len(cols_to_process)
                and len(new_datetime_columns)
                and scale_method == Scale.SCALE_NORMAL
            ):
                cols_to_process += new_datetime_columns
            if (
                len(cols_to_process)
                and len(new_text_columns)
                and scale_method == Scale.SCALE_NORMAL
            ):
                cols_to_process += new_text_columns

            if (
                len(cols_to_process)
                and len(golden_columns)
                and scale_method == Scale.SCALE_NORMAL
            ):
                cols_to_process += golden_columns

            if (
                len(cols_to_process)
                and len(kmeans_columns)
                and scale_method == Scale.SCALE_NORMAL
            ):
                cols_to_process += kmeans_columns

            if len(cols_to_process):
                scale = Scale(cols_to_process)
                scale.fit(X_train)
                X_train = scale.transform(X_train)
                self._scale += [scale]

        if self._add_random_feature:
            # -1, 1, with 0 mean
            X_train["random_feature"] = np.random.rand(X_train.shape[0]) * 2.0 - 1.0

        if self._drop_features:
            available_cols = X_train.columns.tolist()
            drop_cols = [c for c in self._drop_features if c in available_cols]
            if len(drop_cols) == X_train.shape[1]:
                raise AutoMLException(
                    "All features are droppped! Your data looks like random data."
                )
            if drop_cols:
                X_train.drop(drop_cols, axis=1, inplace=True)
            self._drop_features = drop_cols

        if X_train is not None:
            # there can be catagorical columns (in CatBoost) which cant be clipped
            numeric_cols = X_train.select_dtypes(include="number").columns.tolist()
            X_train[numeric_cols] = X_train[numeric_cols].clip(
                lower=np.finfo(np.float32).min + 1000,
                upper=np.finfo(np.float32).max - 1000,
            )

        return X_train, y_train, sample_weight

    def transform(self, X_validation, y_validation, sample_weight_validation=None):
        logger.debug("Preprocessing.transform")

        # doing copy to avoid SettingWithCopyWarning
        if X_validation is not None:
            X_validation = X_validation.copy(deep=False)
        if y_validation is not None:
            y_validation = y_validation.copy(deep=False)

        # target preprocessing
        # this must be used first, maybe we will drop some rows because of missing target values
        if y_validation is not None:
            target_preprocessing = self._params.get("target_preprocessing")
            logger.debug("target_preprocessing -> {}".format(target_preprocessing))

            X_validation, y_validation, sample_weight_validation = ExcludeRowsMissingTarget.transform(
                X_validation, y_validation, sample_weight_validation
            )

            if PreprocessingCategorical.CONVERT_INTEGER in target_preprocessing:
                if y_validation is not None and self._categorical_y is not None:
                    y_validation = pd.Series(
                        self._categorical_y.transform(y_validation)
                    )

            if PreprocessingCategorical.CONVERT_ONE_HOT in target_preprocessing:
                if y_validation is not None and self._categorical_y is not None:
                    y_validation = self._categorical_y.transform(
                        pd.DataFrame({"target": y_validation}), "target"
                    )

            if Scale.SCALE_LOG_AND_NORMAL in target_preprocessing:
                if self._scale_y is not None and y_validation is not None:
                    logger.debug("Transform log and normalize")
                    y_validation = pd.DataFrame({"target": y_validation})
                    y_validation = self._scale_y.transform(y_validation)
                    y_validation = y_validation["target"]

            if Scale.SCALE_NORMAL in target_preprocessing:
                if self._scale_y is not None and y_validation is not None:
                    logger.debug("Transform normalize")
                    y_validation = pd.DataFrame({"target": y_validation})
                    y_validation = self._scale_y.transform(y_validation)
                    y_validation = y_validation["target"]

        # columns preprocessing
        if len(self._remove_columns) and X_validation is not None:
            cols_to_remove = [
                col for col in X_validation.columns if col in self._remove_columns
            ]
            X_validation.drop(cols_to_remove, axis=1, inplace=True)

        # text transform
        for tt in self._text_transforms:
            if X_validation is not None and tt is not None:
                X_validation = tt.transform(X_validation)

        for missing in self._missing_values:
            if X_validation is not None and missing is not None:
                X_validation = missing.transform(X_validation)

        # to be sure that all missing are filled
        # in case new data there can be gaps!
        if (
            X_validation is not None
            and np.sum(np.sum(pd.isnull(X_validation))) > 0
            and len(self._params["columns_preprocessing"]) > 0
        ):
            # there is something missing, fill it
            # we should notice user about it!
            # warnings should go to the separate file ...
            # warnings.warn(
            #    "There are columns {} with missing values which didnt have missing values in train dataset.".format(
            #        list(
            #            X_validation.columns[np.where(np.sum(pd.isnull(X_validation)))]
            #        )
            #    )
            # )
            missing = PreprocessingMissingValues(
                X_validation.columns, PreprocessingMissingValues.FILL_NA_MEDIAN
            )
            missing.fit(X_validation)
            X_validation = missing.transform(X_validation)

        # golden features
        if self._golden_features is not None:
            X_validation = self._golden_features.transform(X_validation)

        if self._kmeans is not None:
            X_validation = self._kmeans.transform(X_validation)

        for convert in self._categorical:
            if X_validation is not None and convert is not None:
                X_validation = convert.transform(X_validation)

        for dtt in self._datetime_transforms:
            if X_validation is not None and dtt is not None:
                X_validation = dtt.transform(X_validation)

        for scale in self._scale:
            if X_validation is not None and scale is not None:
                X_validation = scale.transform(X_validation)

        if self._add_random_feature:
            # -1, 1, with 0 mean
            X_validation["random_feature"] = (
                np.random.rand(X_validation.shape[0]) * 2.0 - 1.0
            )

        if self._drop_features and X_validation is not None:
            X_validation.drop(self._drop_features, axis=1, inplace=True)

        if X_validation is not None:
            # there can be catagorical columns (in CatBoost) which cant be clipped
            numeric_cols = X_validation.select_dtypes(include="number").columns.tolist()
            X_validation[numeric_cols] = X_validation[numeric_cols].clip(
                lower=np.finfo(np.float32).min + 1000,
                upper=np.finfo(np.float32).max - 1000,
            )

        return X_validation, y_validation, sample_weight_validation

    def inverse_scale_target(self, y):
        if self._scale_y is not None:
            y = pd.DataFrame({"target": y})
            y = self._scale_y.inverse_transform(y)
            y = y["target"]
        return y

    def inverse_categorical_target(self, y):
        if self._categorical_y is not None:
            y = self._categorical_y.inverse_transform(
                pd.DataFrame({"target": np.array(y)})
            )
            y = y.astype(str)
        return y

    def get_target_class_names(self):
        pos_label, neg_label = "1", "0"
        if self._categorical_y is not None:
            if self._params["ml_task"] == BINARY_CLASSIFICATION:
                # binary classification
                for label, value in self._categorical_y.to_json().items():
                    if value == 1:
                        pos_label = label
                    else:
                        neg_label = label
                return [neg_label, pos_label]
            else:
                # multiclass classification
                # logger.debug(self._categorical_y.to_json())
                if "unique_values" not in self._categorical_y.to_json():
                    labels = dict(
                        (v, k) for k, v in self._categorical_y.to_json().items()
                    )
                else:
                    labels = {
                        i: v
                        for i, v in enumerate(
                            self._categorical_y.to_json()["unique_values"]
                        )
                    }

                return list(labels.values())

        else:  # self._categorical_y is None
            if "ml_task" in self._params:
                if self._params["ml_task"] == BINARY_CLASSIFICATION:
                    return ["0", "1"]
        return []

    def prepare_target_labels(self, y):
        pos_label, neg_label = "1", "0"

        if self._categorical_y is not None:
            if len(y.shape) == 1:
                # binary classification
                for label, value in self._categorical_y.to_json().items():
                    if value == 1:
                        pos_label = label
                    else:
                        neg_label = label
                # threshold is applied in AutoML class
                return pd.DataFrame(
                    {
                        "prediction_{}".format(neg_label): 1 - y,
                        "prediction_{}".format(pos_label): y,
                    }
                )
            else:
                # multiclass classification
                if "unique_values" not in self._categorical_y.to_json():
                    labels = dict(
                        (v, k) for k, v in self._categorical_y.to_json().items()
                    )
                else:
                    labels = {
                        i: v
                        for i, v in enumerate(
                            self._categorical_y.to_json()["unique_values"]
                        )
                    }

                d = {}
                cols = []
                for i in range(y.shape[1]):
                    d["prediction_{}".format(labels[i])] = y[:, i]
                    cols += ["prediction_{}".format(labels[i])]
                df = pd.DataFrame(d)
                df["label"] = np.argmax(np.array(df[cols]), axis=1)

                df["label"] = df["label"].map(labels)

                return df
        else:  # self._categorical_y is None
            if "ml_task" in self._params:
                if self._params["ml_task"] == BINARY_CLASSIFICATION:
                    return pd.DataFrame({"prediction_0": 1 - y, "prediction_1": y})
                elif self._params["ml_task"] == MULTICLASS_CLASSIFICATION:
                    return pd.DataFrame(
                        data=y,
                        columns=["prediction_{}".format(i) for i in range(y.shape[1])],
                    )

        return pd.DataFrame({"prediction": y})

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

        if self._datetime_transforms is not None and len(self._datetime_transforms):
            dtts = []
            for dtt in self._datetime_transforms:
                dtts += [dtt.to_json()]
            if dtts:
                preprocessing_params["datetime_transforms"] = dtts

        if self._text_transforms is not None and len(self._text_transforms):
            tts = []
            for tt in self._text_transforms:
                tts += [tt.to_json()]
            if tts:
                preprocessing_params["text_transforms"] = tts

        if self._golden_features is not None:
            preprocessing_params["golden_features"] = self._golden_features.to_json()

        if self._kmeans is not None:
            preprocessing_params["kmeans"] = self._kmeans.to_json()

        if self._scale is not None and len(self._scale):
            scs = [sc.to_json() for sc in self._scale if sc.to_json()]
            if scs:
                preprocessing_params["scale"] = scs
        if self._categorical_y is not None:
            cat_y = self._categorical_y.to_json()
            if cat_y:
                preprocessing_params["categorical_y"] = cat_y
        if self._scale_y is not None:
            preprocessing_params["scale_y"] = self._scale_y.to_json()

        if "ml_task" in self._params:
            preprocessing_params["ml_task"] = self._params["ml_task"]

        if self._add_random_feature:
            preprocessing_params["add_random_feature"] = True

        if self._drop_features:
            preprocessing_params["drop_features"] = self._drop_features

        preprocessing_params["params"] = self._params

        return preprocessing_params

    def from_json(self, data_json):

        self._params = data_json.get("params", self._params)

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

        if "datetime_transforms" in data_json:
            self._datetime_transforms = []
            for dtt_params in data_json["datetime_transforms"]:
                dtt = DateTimeTransformer()
                dtt.from_json(dtt_params)
                self._datetime_transforms += [dtt]

        if "text_transforms" in data_json:
            self._text_transforms = []
            for tt_params in data_json["text_transforms"]:
                tt = TextTransformer()
                tt.from_json(tt_params)
                self._text_transforms += [tt]

        if "golden_features" in data_json:
            self._golden_features = GoldenFeaturesTransformer()
            self._golden_features.from_json(data_json["golden_features"])

        if "kmeans" in data_json:
            self._kmeans = KMeansTransformer()
            self._kmeans.from_json(data_json["kmeans"])

        if "scale" in data_json:
            self._scale = []
            for scale_data in data_json["scale"]:
                sc = Scale()
                sc.from_json(scale_data)
                self._scale += [sc]
        if "categorical_y" in data_json:
            if "new_columns" in data_json["categorical_y"]:
                self._categorical_y = LabelBinarizer()
            else:
                self._categorical_y = LabelEncoder()

            self._categorical_y.from_json(data_json["categorical_y"])
        if "scale_y" in data_json:
            self._scale_y = Scale()
            self._scale_y.from_json(data_json["scale_y"])
        if "ml_task" in data_json:
            self._params["ml_task"] = data_json["ml_task"]

        self._add_random_feature = data_json.get("add_random_feature", False)
        self._drop_features = data_json.get("drop_features", [])
