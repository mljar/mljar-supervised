"""Optional pytabkit adapter. Imported only when RealMLP is used.

Capture upstream epoch metrics without redirecting stdout or monkey-patching
global Lightning classes. Keep this small adapter covered against pytabkit 1.7.
"""

import re
import warnings

from pytabkit import RealMLP_TD_Classifier, RealMLP_TD_Regressor
from pytabkit.models.alg_interfaces.nn_interfaces import NNAlgInterface

from supervised.algorithms._pytabkit_compat import PytabkitCompatibilityMixin


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

    def predict(self, ds):
        # pytabkit 1.7.2 omits the fit-time DummyLogger when constructing its
        # prediction Trainer, unintentionally enabling Lightning's CSV logger.
        from pytorch_lightning.loggers.logger import DummyLogger

        self.min_trainer_kwargs["logger"] = DummyLogger()
        with warnings.catch_warnings():
            if str(self.device) == "cpu":
                # Portable RealMLP artifacts intentionally reload on CPU. This
                # is a supported inference path, not an accidental GPU fallback.
                warnings.filterwarnings(
                    "ignore",
                    message=(
                        r"GPU available but not used\. You can set it by doing "
                        r"`Trainer\(accelerator='gpu'\)`\.\Z"
                    ),
                    category=UserWarning,
                    module=r"pytorch_lightning\.trainer\.setup\Z",
                )
            return super().predict(ds)


class HistoryMixin(PytabkitCompatibilityMixin):
    def _create_alg_interface(self, n_cv):
        return HistoryInterface(**self.get_config())


class RealMLPClassifier(HistoryMixin, RealMLP_TD_Classifier):
    pass


class RealMLPRegressor(HistoryMixin, RealMLP_TD_Regressor):
    pass
