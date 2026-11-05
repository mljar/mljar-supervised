"""Optional pytabkit adapter. Imported only when RealMLP is used.

Capture upstream epoch metrics without redirecting stdout or monkey-patching
global Lightning classes. Keep this small adapter covered against pytabkit 1.7.
"""

import re
import warnings
import os
from pathlib import Path

from pytabkit import RealMLP_TD_Classifier, RealMLP_TD_Regressor
from pytabkit.models.alg_interfaces.nn_interfaces import NNAlgInterface

from supervised.algorithms._pytabkit_compat import PytabkitCompatibilityMixin


class HistoryLogger:
    def __init__(self, logger, history, on_epoch=None):
        self.logger = logger
        self.history = history
        self.on_epoch = on_epoch

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
            if self.on_epoch is not None:
                self.on_epoch(int(match[1]))
        self.logger.log(verbosity, content)

    def force_log(self, content):
        self.logger.force_log(content)


class HistoryInterface(NNAlgInterface):
    def capture_checkpoint(self, epoch):
        # The upstream epoch log is emitted during validation, before restoring
        # the individual best weights. Capture tensors, not rounded log scores.
        if epoch not in self.checkpoint_epochs:
            return
        import torch

        module = self.model
        if epoch != module.progress.epoch + 1:
            raise RuntimeError("Unexpected RealMLP validation checkpoint timing")
        state = {
            kind: {name: value.detach().cpu().clone() for name, value in values}
            for kind, values in (
                ("parameters", module.model.named_parameters()),
                ("buffers", module.model.named_buffers()),
            )
        }
        path = Path(self.checkpoint_directory) / f"epoch_{epoch:04d}.pt"
        temporary = path.with_suffix(".tmp")
        torch.save(state, temporary)
        os.replace(temporary, path)

    def fit(self, ds, idxs_list, interface_resources, logger, tmp_folders, name):
        self.history = []
        capture = self.capture_checkpoint if getattr(self, "checkpoint_directory", None) else None
        history_logger = HistoryLogger(logger, self.history, capture)
        try:
            return super().fit(
                ds, idxs_list, interface_resources, history_logger, tmp_folders, name,
            )
        finally:
            # The fitted module retains its logger. Do not serialize callbacks,
            # circular interface references, or temporary checkpoint paths.
            history_logger.on_epoch = None
            self.checkpoint_directory = None

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
        interface = HistoryInterface(**self.get_config())
        interface.checkpoint_directory = getattr(self, "_checkpoint_directory", None)
        interface.checkpoint_epochs = getattr(self, "_checkpoint_epochs", ())
        return interface


def restore_checkpoint(estimator, path):
    """Restore inference weights while retaining this fold's fitted transforms."""
    import torch

    interface = estimator.cv_alg_interface_
    if "val_metric_name" in interface.get_current_predict_params_dict():
        raise RuntimeError("Shared RealMLP selection requires a single validation metric")
    network = interface.model.model
    state = torch.load(path, map_location="cpu", weights_only=True)
    with torch.no_grad():
        for kind, values in (
            ("parameters", network.named_parameters()),
            ("buffers", network.named_buffers()),
        ):
            values = dict(values)
            if values.keys() != state[kind].keys():
                raise RuntimeError("RealMLP checkpoint does not match this fold's network")
            for name, value in values.items():
                saved = state[kind][name]
                if value.shape != saved.shape or value.dtype != saved.dtype:
                    raise RuntimeError("RealMLP checkpoint tensor shape or dtype changed")
                value.copy_(saved)


class RealMLPClassifier(HistoryMixin, RealMLP_TD_Classifier):
    pass


class RealMLPRegressor(HistoryMixin, RealMLP_TD_Regressor):
    pass
