import numpy as np
from sklearn import preprocessing

class Scaler(object):
    def __init__(self, columns = []):
        self.scale = preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True)
        self.columns = columns

    def fit(self, X):
        self.scale.fit(X) #np.array(x).reshape(-1, 1))

    def transform(self, X):
        return self.scale.transform(list(x.values))

    def to_json(self):
        data_json = {"scale":self.scale.scale_,
            "mean":self.scale.mean_,
            "var": self.scale.var_,
            "n_samples_seen":self.scale.n_samples_seen_}
        return data_json

    def from_json(self, data_json):
        self.scale = preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True)
        self.scale.scale_ = data_json.get("scale")
        self.scale.mean_ = data_json.get("mean")
        self.scale.var_ = data_json.get("var")
        self.scale.n_samples_seen_ = data_json.get("n_samples_seen")
