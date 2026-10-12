import logging
from decimal import Decimal

import numpy as np
from sklearn import preprocessing as sk_preproc

from supervised.utils.config import LOG_LEVEL

logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)


class LabelEncoder(object):
    def __init__(self, try_to_fit_numeric=False):
        self.lbl = sk_preproc.LabelEncoder()
        self._try_to_fit_numeric = try_to_fit_numeric

    def fit(self, x):
        values = np.asarray(x)
        if values.size and all(
            isinstance(value, (bool, np.bool_)) for value in values.flat
        ):
            # Avoid sklearn converting pandas extension booleans to floats.
            x = values.astype(object)
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

    def _normalize_booleans(self, x):
        values = np.asarray(x)
        classes = getattr(self.lbl, "classes_", None)
        if classes is None:
            return x
        if classes.size and all(
            isinstance(value, (bool, np.bool_)) for value in classes
        ):
            if any(
                isinstance(value, str) and value not in ("False", "True")
                for value in values.flat
            ):
                # A legacy mapping also represents literal "False"/"True"
                # string categories. Preserve their unseen-string behavior.
                self.lbl.classes_ = np.array(
                    [str(value) for value in classes], dtype=object
                )
                return np.array(
                    [str(value) for value in values.flat], dtype=object
                ).reshape(values.shape)
            normalized = []
            for value in values.flat:
                if isinstance(value, (bool, np.bool_)):
                    normalized.append(bool(value))
                elif isinstance(value, str) and value in ("False", "True"):
                    normalized.append(value == "True")
                elif isinstance(value, (int, float, np.number)) and value in (0, 1):
                    normalized.append(bool(value))
                else:
                    raise ValueError(f"Invalid boolean label: {value!r}")
            # Object lookup preserves codes when an unseen boolean is appended
            # after a single-class fold, rather than assuming numeric sort order.
            self.lbl.classes_ = classes.astype(object)
            return np.array(normalized, dtype=object).reshape(values.shape)
        if values.size and all(
            isinstance(value, (bool, np.bool_)) for value in values.flat
        ):
            if all(isinstance(value, str) for value in classes):
                keys = set(classes)
                if keys & {"False", "True"}:
                    normalized = [str(bool(value)) for value in values.flat]
                elif keys and keys <= {"0", "1", "0.0", "1.0"}:
                    # Legacy extension-boolean encoders serialized float labels.
                    float_labels = any("." in key for key in keys)
                    normalized = [
                        str(float(value)) if float_labels else str(int(value))
                        for value in values.flat
                    ]
                else:
                    return x
                self.lbl.classes_ = classes.astype(object)
                return np.array(normalized, dtype=object).reshape(values.shape)
        return x

    def transform(self, x):
        x = self._normalize_booleans(x)
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
        # JSON key order must not change the stored label codes.
        keys = np.array(sorted(data_json, key=data_json.get))
        if len(keys) == 2 and "False" in keys and "True" in keys:
            keys = np.array([key == "True" for key in keys])
        self.lbl.classes_ = keys
