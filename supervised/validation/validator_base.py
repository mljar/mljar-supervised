import logging

from supervised.utils.utils import load_data

log = logging.getLogger(__name__)


class BaseValidator(object):

    def __init__(self, params, data_store=None):
        self.params = params
        self._data_store = data_store

    def _load_rows(self, indices, *, positional=True):
        data = {}
        for name, path, column in (
            ("X", self._X_path, None),
            ("y", self._y_path, "target"),
            ("sample_weight", self._sample_weight_path, "sample_weight"),
            ("sensitive_features", self._sensitive_features_path, None),
        ):
            if path is not None:
                selected = load_data(
                    path, store=self._data_store, rows=indices, positional=positional
                )
                data[name] = selected if column is None else selected[column]
        return data

    def split(self):
        pass

    def get_n_splits(self):
        pass

    def get_repeats(self):
        return 1
