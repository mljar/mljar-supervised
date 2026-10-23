import numpy as np
import pandas as pd

from supervised.preprocessing.label_binarizer import LabelBinarizer
from supervised.preprocessing.label_encoder import LabelEncoder


class PreprocessingCategorical(object):
    CONVERT_ONE_HOT = "categorical_to_onehot"
    CONVERT_INTEGER = "categorical_to_int"

    FEW_CATEGORIES = "few_categories"
    MANY_CATEGORIES = "many_categories"

    def __init__(self, columns=[], method=CONVERT_INTEGER):
        self._convert_method = method
        self._convert_params = {}
        self._columns = columns
        self._enc = None

    def fit(self, X, y=None):
        self._fit_categorical_convert(X)

    def _fit_categorical_convert(self, X):
        for column in self._columns:
            dtype = X[column].dtype
            if (
                pd.api.types.is_numeric_dtype(dtype)
                and not pd.api.types.is_bool_dtype(dtype)
            ) or pd.api.types.is_datetime64_any_dtype(dtype):
                # Numeric/datetime inputs are already represented by their dtype.
                continue
            # The preprocessing plan already selected categorical encoding.
            # Do not rerun text detection on a fold: its cardinality ratio can
            # differ from the full data and otherwise leave raw strings (#147).
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
        if not self._convert_params:
            return X
        columns = dict(X.items())
        for column, lbl_params in self._convert_params.items():
            if "unique_values" in lbl_params and "new_columns" in lbl_params:
                # convert to one hot
                lbl = LabelBinarizer()
                lbl.from_json(lbl_params)
                columns.update(lbl._encoded_columns(columns[column], column))
                del columns[column]
            else:
                # convert to integer
                lbl = LabelEncoder()
                lbl.from_json(lbl_params)
                # All unseen feature values share a fixed code after the fitted
                # classes. Their encoding cannot depend on batch size or order.
                transformed_values = lbl.transform(
                    columns[column], unknown_value=len(lbl_params)
                )
                # Replace the column so boolean/string/category extension dtypes
                # cannot reject encoded integer values (#333).
                columns[column] = transformed_values

        return pd.DataFrame(columns, index=X.index)

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
                transformed_values = lbl.inverse_transform(X.loc[:, column])
                # check for pandas FutureWarning: Setting an item
                # of incompatible dtype is deprecated and will raise
                # in a future error of pandas.
                if transformed_values.dtype != X.loc[:, column].dtype and \
                        (X.loc[:, column].dtype == bool or X.loc[:, column].dtype == int):
                        X = X.astype({column: transformed_values.dtype})
                X.loc[:, column] = transformed_values

        return X

    def to_json(self):
        params = {}
        
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
            self._convert_params = params.get("convert_params", {})

        else:
            self._convert_method, self._convert_params = None, None
            self._columns = []
