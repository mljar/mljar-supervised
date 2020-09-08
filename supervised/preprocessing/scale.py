import pandas as pd
import numpy as np

from sklearn import preprocessing


class Scale(object):

    SCALE_NORMAL = "scale_normal"
    SCALE_LOG_AND_NORMAL = "scale_log_and_normal"

    def __init__(self, columns=[], scale_method=SCALE_NORMAL):
        self.scale_method = scale_method
        self.columns = columns
        self.scale = preprocessing.StandardScaler(
            copy=True, with_mean=True, with_std=True
        )
        self.X_min_values = None  # it is used in SCALE_LOG_AND_NORMAL

    def fit(self, X):

        if len(self.columns):
            for c in self.columns:
                X[c] = X[c].astype(float)

            if self.scale_method == self.SCALE_NORMAL:
                self.scale.fit(X[self.columns])
            elif self.scale_method == self.SCALE_LOG_AND_NORMAL:
                self.X_min_values = np.min(X[self.columns])
                self.scale.fit(np.log(X[self.columns] - self.X_min_values + 1))

    def transform(self, X):

        if len(self.columns):
            X.loc[:, self.columns] = X.loc[:, self.columns].astype(float)
            if self.scale_method == self.SCALE_NORMAL:
                X.loc[:, self.columns] = self.scale.transform(X[self.columns])
            elif self.scale_method == self.SCALE_LOG_AND_NORMAL:

                X[self.columns] = np.log(
                    np.clip(
                        X[self.columns] - self.X_min_values + 1, a_min=1, a_max=None
                    )
                )
                X.loc[:, self.columns] = self.scale.transform(X[self.columns])
        return X

    def inverse_transform(self, X):

        if len(self.columns):

            if self.scale_method == self.SCALE_NORMAL:
                X.loc[:, self.columns] = self.scale.inverse_transform(X[self.columns])
            elif self.scale_method == self.SCALE_LOG_AND_NORMAL:

                X[self.columns] = self.scale.inverse_transform(X[self.columns])
                X[self.columns] = np.exp(X[self.columns])

                X.loc[:, self.columns] += self.X_min_values - 1
        return X

    def to_json(self):

        if len(self.columns) == 0:
            return None
        data_json = {
            "scale": list(self.scale.scale_),
            "mean": list(self.scale.mean_),
            "var": list(self.scale.var_),
            "n_samples_seen": int(self.scale.n_samples_seen_),
            "n_features_in": int(self.scale.n_features_in_),
            "columns": self.columns,
            "scale_method": self.scale_method,
        }
        if self.X_min_values is not None:
            data_json["X_min_values"] = list(self.X_min_values)
        return data_json

    def from_json(self, data_json):
        self.scale = preprocessing.StandardScaler(
            copy=True, with_mean=True, with_std=True
        )
        self.scale.scale_ = data_json.get("scale")
        if self.scale.scale_ is not None:
            self.scale.scale_ = np.array(self.scale.scale_)
        self.scale.mean_ = data_json.get("mean")
        if self.scale.mean_ is not None:
            self.scale.mean_ = np.array(self.scale.mean_)
        self.scale.var_ = data_json.get("var")
        if self.scale.var_ is not None:
            self.scale.var_ = np.array(self.scale.var_)
        self.scale.n_samples_seen_ = int(data_json.get("n_samples_seen"))
        self.scale.n_features_in_ = int(data_json.get("n_features_in"))
        self.columns = data_json.get("columns", [])
        self.scale_method = data_json.get("scale_method")
        self.X_min_values = data_json.get("X_min_values")
        if self.X_min_values is not None:
            self.X_min_values = np.array(self.X_min_values)
