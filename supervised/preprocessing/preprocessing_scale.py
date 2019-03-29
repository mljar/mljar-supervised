import pandas as pd
import numpy as np

from sklearn import preprocessing


class PreprocessingScale(object):

    SCALE_MIN_MAX = "scale_min_max"
    SCALE_NORMAL = "scale_normal"
    SCALE_LOG = "scale_log"
    SCALE_LOG_AND_NORMAL = "scale_log_and_normal"

    def __init__(self, columns=[]):
        self.scale = preprocessing.StandardScaler(
            copy=True, with_mean=True, with_std=True
        )
        self.columns = columns

    def fit(self, X):
        if len(self.columns):
            self.scale.fit(X[self.columns])

    def transform(self, X):
        if len(self.columns):
            X.loc[:, self.columns] = self.scale.transform(X[self.columns])
        return X

    def to_json(self):
        if ~len(self.columns):
            return None
        data_json = {
            "scale": self.scale.scale_,
            "mean": self.scale.mean_,
            "var": self.scale.var_,
            "n_samples_seen": self.scale.n_samples_seen_,
            "columns": self.columns,
        }
        return data_json

    def from_json(self, data_json):
        self.scale = preprocessing.StandardScaler(
            copy=True, with_mean=True, with_std=True
        )
        self.scale.scale_ = data_json.get("scale")
        self.scale.mean_ = data_json.get("mean")
        self.scale.var_ = data_json.get("var")
        self.scale.n_samples_seen_ = data_json.get("n_samples_seen")
        self.columns = data_json.get("columns", [])
