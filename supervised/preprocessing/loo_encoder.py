import copy
import json
import logging
import numpy as np
import pandas as pd
import warnings
from decimal import Decimal
from category_encoders.leave_one_out import LeaveOneOutEncoder

from supervised.utils.config import LOG_LEVEL

logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)


class LooEncoder(object):
    def __init__(self, cols=None):
        self.enc = LeaveOneOutEncoder(
            cols=cols,
            verbose=1,
            drop_invariant=False,
            return_df=True,
            handle_unknown="value",
            handle_missing="value",
            random_state=1,
            sigma=0,
        )

    def fit(self, X, y):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.enc.fit(X, y)

    def transform(self, X):
        return self.enc.transform(X)

    def to_json(self):
        data_json = {
            "cols": self.enc.cols,
            "dim": self.enc._dim,
            "mean": float(self.enc._mean),
            "feature_names": self.enc.feature_names,
            "mapping": {},
        }
        for k, v in self.enc.mapping.items():
            data_json["mapping"][k] = v.to_json()
        return data_json

    def from_json(self, data_json):
        self.enc.cols = data_json.get("cols")
        self.enc._dim = data_json.get("dim")
        self.enc._mean = data_json.get("mean")
        self.enc.feature_names = data_json.get("feature_names")
        self.enc.mapping = {}
        for k, v in data_json.get("mapping", {}).items():
            self.enc.mapping[k] = pd.DataFrame(json.loads(v))
