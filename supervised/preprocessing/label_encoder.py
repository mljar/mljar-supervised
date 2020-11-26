import copy
import logging
import numpy as np
from decimal import Decimal
from sklearn import preprocessing as sk_preproc

from supervised.utils.config import LOG_LEVEL

logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)


class LabelEncoder(object):
    def __init__(self, try_to_fit_numeric=False):
        self.lbl = sk_preproc.LabelEncoder()
        self._try_to_fit_numeric = try_to_fit_numeric

    def fit(self, x):
        self.lbl.fit(x)  # list(x.values))
        if self._try_to_fit_numeric:
            logger.debug("Try to fit numeric in LabelEncoder")
            try:
                arr = {Decimal(c): c for c in self.lbl.classes_}
                sorted_arr = dict(sorted(arr.items()))
                self.lbl.classes_ = np.array(
                    list(sorted_arr.values()), dtype=self.lbl.classes_.dtype
                )
            except Exception as e:
                pass

    def transform(self, x):
        try:
            return self.lbl.transform(x)  # list(x.values))
        except ValueError as ve:
            # rescue
            classes = np.unique(x)  # list(x.values))
            diff = np.setdiff1d(classes, self.lbl.classes_)
            self.lbl.classes_ = np.concatenate((self.lbl.classes_, diff))
            return self.lbl.transform(x)  # list(x.values))

    def inverse_transform(self, x):
        return self.lbl.inverse_transform(x)  # (list(x.values))

    def to_json(self):
        data_json = {}
        for i, cl in enumerate(self.lbl.classes_):
            data_json[str(cl)] = i
        return data_json

    def from_json(self, data_json):
        keys = np.array(list(data_json.keys()))
        if len(keys) == 2 and "False" in keys and "True" in keys:
            keys = [False, True]
        self.lbl.classes_ = keys
