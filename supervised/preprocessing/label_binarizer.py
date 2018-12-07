import numpy as np
import json


class LabelBinarizer(object):
    def __init__(self):
        self._new_columns = []
        self._uniq_values = None

    def fit(self, X, column):
        self._uniq_values = np.unique(X[column].values)

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

    def to_json(self):
        self._uniq_values = [
            i if type(i) != np.bool_ else bool(i) for i in list(self._uniq_values)
        ]
        data_json = {
            "new_columns": list(self._new_columns),
            "unique_values": self._uniq_values,
        }
        return data_json

    def from_json(self, data_json):
        self._new_columns = data_json.get("new_columns", None)
        self._uniq_values = data_json.get("unique_values", None)
