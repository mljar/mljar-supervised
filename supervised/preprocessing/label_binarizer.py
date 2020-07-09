import numpy as np
import json


class LabelBinarizer(object):
    def __init__(self):
        self._new_columns = []
        self._uniq_values = None
        self._old_column = None

    def fit(self, X, column):
        self._old_column = column
        self._uniq_values = np.unique(X[column].values)
        # self._uniq_values = [str(u) for u in self._uniq_values]

        if len(self._uniq_values) == 2:
            self._new_columns.append(column + "_" + str(self._uniq_values[1]))
        else:
            for v in self._uniq_values:
                self._new_columns.append(column + "_" + str(v))

    def transform(self, X, column):
        if len(self._uniq_values) == 2:
            X[column + "_" + str(self._uniq_values[1])] = (
                X[column] == self._uniq_values[1]
            ).astype(int)
        else:
            for v in self._uniq_values:
                X[column + "_" + str(v)] = (X[column] == v).astype(int)

        X.drop(column, axis=1, inplace=True)
        return X

    def inverse_transform(self, X):
        if self._old_column is None:
            return X

        old_col = X[self._new_columns[0]] * 0

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
        self._uniq_values = [
            i if type(i) != np.bool_ else bool(i) for i in list(self._uniq_values)
        ]
        data_json = {
            "new_columns": list(self._new_columns),
            "unique_values": self._uniq_values,
            "old_column": self._old_column,
        }

        return data_json

    def from_json(self, data_json):
        self._new_columns = data_json.get("new_columns", None)
        self._uniq_values = data_json.get("unique_values", None)
        self._old_column = data_json.get("old_column", None)
