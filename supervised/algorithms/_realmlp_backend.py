"""Optional pytabkit adapter. Imported only when RealMLP is used.

Capture upstream epoch metrics without redirecting stdout or monkey-patching
global Lightning classes. Keep this small adapter covered against pytabkit 1.7.
"""

import re

from pytabkit import RealMLP_TD_Classifier, RealMLP_TD_Regressor
from pytabkit.models.alg_interfaces.nn_interfaces import NNAlgInterface


class HistoryLogger:
    def __init__(self, logger, history):
        self.logger = logger
        self.history = history

    def get_verbosity_level(self):
        return self.logger.get_verbosity_level()

    def log(self, verbosity, content):
        match = re.match(r"Epoch (\d+)/(\d+): val (\S+) =\s*(\S+)", content)
        if match:
            self.history.append(
                {
                    "epoch": int(match[1]),
                    "metric": match[3],
                    "validation": float(match[4]),
                }
            )
        self.logger.log(verbosity, content)

    def force_log(self, content):
        self.logger.force_log(content)


class HistoryInterface(NNAlgInterface):
    def fit(self, ds, idxs_list, interface_resources, logger, tmp_folders, name):
        self.history = []
        return super().fit(
            ds,
            idxs_list,
            interface_resources,
            HistoryLogger(logger, self.history),
            tmp_folders,
            name,
        )


class HistoryMixin:
    def _create_alg_interface(self, n_cv):
        return HistoryInterface(**self.get_config())


class RealMLPClassifier(HistoryMixin, RealMLP_TD_Classifier):
    pass


class RealMLPRegressor(HistoryMixin, RealMLP_TD_Regressor):
    pass
