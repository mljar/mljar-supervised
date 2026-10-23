import numpy as np
import pandas as pd


class LabelBinarizer(object):
    def __init__(self):
        self._new_columns = []
        self._uniq_values = None
        self._old_column = None
        self._old_column_dtype = None

    def fit(self, X, column):
        self._old_column = column
        self._old_column_dtype = str(X[column].dtype)
        self._uniq_values = np.unique(X[column].values)
        # self._uniq_values = [str(u) for u in self._uniq_values]

        if len(self._uniq_values) == 2:
            self._new_columns.append(column + "_" + str(self._uniq_values[1]))
        else:
            for v in self._uniq_values:
                self._new_columns.append(column + "_" + str(v))

    def _encoded_columns(self, values, column):
        unique = self._uniq_values[1:] if len(self._uniq_values) == 2 else self._uniq_values
        return {f"{column}_{v}": (values == v).astype(int) for v in unique}

    def transform(self, X, column):
        # Construct the frame once, retaining assignment order (including an
        # existing column overwritten by an encoded name) without fragmentation.
        columns = dict(X.items())
        columns.update(self._encoded_columns(columns[column], column))
        del columns[column]
        return pd.DataFrame(columns, index=X.index)

    def inverse_transform(self, X):
        if self._old_column is None:
            return X

        old_col = (X[self._new_columns[0]] * 0).astype(self._old_column_dtype)

        for unique_value in self._uniq_values:
            new_col = f"{self._old_column}_{unique_value}"
            if new_col not in self._new_columns:
                old_col[:] = unique_value
            else:
                old_col[X[new_col] == 1] = unique_value

        X[self._old_column] = old_col
        X.drop(self._new_columns, axis=1, inplace=True)
        return X

    def to_json(self):
        self._uniq_values = [str(i) for i in list(self._uniq_values)]
        data_json = {
            "new_columns": list(self._new_columns),
            "unique_values": self._uniq_values,
            "old_column": self._old_column,
            "old_column_dtype": self._old_column_dtype,
        }

        if (
            "True" in self._uniq_values
            and "False" in self._uniq_values
            and len(self._uniq_values) == 2
        ):
            self._uniq_values = [False, True]

        return data_json

    def from_json(self, data_json):
        self._new_columns = data_json.get("new_columns", None)
        self._uniq_values = data_json.get("unique_values", None)
        self._old_column = data_json.get("old_column", None)
        self._old_column_dtype = data_json.get("old_column_dtype", None)

        if (
            "True" in self._uniq_values
            and "False" in self._uniq_values
            and len(self._uniq_values) == 2
        ):
            self._uniq_values = [False, True]
