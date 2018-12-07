import numpy as np
from sklearn import preprocessing as sk_preproc


class LabelEncoder(object):
    def __init__(self):
        self.lbl = sk_preproc.LabelEncoder()

    def fit(self, x):
        self.lbl.fit(list(x.values))

    def transform(self, x):
        try:
            return self.lbl.transform(list(x.values))
        except ValueError as ve:
            # rescue
            classes = np.unique(list(x.values))
            diff = np.setdiff1d(classes, self.lbl.classes_)
            self.lbl.classes_ = np.concatenate((self.lbl.classes_, diff))
            return self.lbl.transform(list(x.values))

    def to_json(self):
        data_json = {}
        for i, cl in enumerate(self.lbl.classes_):
            data_json[str(cl)] = i
        return data_json

    def from_json(self, data_json):
        keys = np.unique(list(data_json.keys()))
        if len(keys) == 2 and "False" in keys and "True" in keys:
            keys = [False, True]
        self.lbl.classes_ = keys
