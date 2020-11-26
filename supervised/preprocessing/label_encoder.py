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
        self.lbl.fit(list(x.values))
        if self._try_to_fit_numeric:
            logger.debug("Try to fit numeric in LabelEncoder")
            classes = copy.deepcopy(self.lbl.classes_)
            try:
                arr = [Decimal(c) for c in classes]
                arr = sorted(arr)
                self.lbl.classes_ = np.array([str(i) for i in arr])
            except Exception as e:
                logger.error(
                    "Eception during try to fit numeric in LabelEncoder, " + str(e)
                )

    def transform(self, x):
        try:
            return self.lbl.transform(list(x.values))
        except ValueError as ve:
            # rescue
            classes = np.unique(list(x.values))
            diff = np.setdiff1d(classes, self.lbl.classes_)
            self.lbl.classes_ = np.concatenate((self.lbl.classes_, diff))
            return self.lbl.transform(list(x.values))

    def inverse_transform(self, x):
        return self.lbl.inverse_transform(list(x.values))

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
