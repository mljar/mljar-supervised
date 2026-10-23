"""Scoped compatibility for the supported pytabkit 1.7 backend.

pytabkit 1.7.2/1.7.3 use the old sklearn finite-check keyword through 1.7.
Lightning 2.6 still constructs the deprecated torch LeafSpec directly.
Use the replacement APIs during our calls, without changing installed packages
or suppressing deprecation warnings. Remove these bridges when upstream fixes
are included in the supported dependency range.
"""

import inspect
import logging
import threading
import warnings
from contextlib import ExitStack, contextmanager

from sklearn.utils.validation import check_array, check_X_y


_backend_lock = threading.RLock()


def _finite_check(check, *args, **kwargs):
    keyword = (
        "ensure_all_finite"
        if "ensure_all_finite" in inspect.signature(check).parameters
        else "force_all_finite"
    )
    old_keyword = (
        "force_all_finite" if keyword == "ensure_all_finite" else "ensure_all_finite"
    )
    if old_keyword in kwargs:
        kwargs[keyword] = kwargs.pop(old_keyword)
    return check(*args, **kwargs)


def _check_X_y(*args, **kwargs):
    return _finite_check(check_X_y, *args, **kwargs)


def _check_array(*args, **kwargs):
    return _finite_check(check_array, *args, **kwargs)


def _leaf_spec():
    from torch.utils._pytree import tree_flatten

    # Let torch construct its current leaf representation (also works on 2.7).
    return tree_flatten(None)[1]


class _BackendLogFilter(logging.Filter):
    def __init__(self):
        super().__init__()
        self.thread = threading.get_ident()

    def filter(self, record):
        # Preserve warnings/errors and logs from unrelated threads.
        return record.thread != self.thread or record.levelno >= logging.WARNING


@contextmanager
def backend_context():
    from pytabkit.models.sklearn import sklearn_base
    from pytorch_lightning.utilities import _pytree

    # Upstream helpers are module globals. Serialize MLJAR backend calls and
    # restore every replacement even if fit/predict raises; nested calls work.
    with _backend_lock, ExitStack() as stack:
        for module, name, replacement in [
            (sklearn_base, "check_X_y_wrapper", _check_X_y),
            (sklearn_base, "check_array_wrapper", _check_array),
            (_pytree, "LeafSpec", _leaf_spec),
        ]:
            original = getattr(module, name)
            stack.callback(setattr, module, name, original)
            setattr(module, name, replacement)

        prefixes = ("pytorch_lightning", "lightning")
        names = set(prefixes) | {
            name
            for name in list(logging.Logger.manager.loggerDict)
            if name.startswith(tuple(prefix + "." for prefix in prefixes))
        }
        log_filter = _BackendLogFilter()
        for name in names:
            logger = logging.getLogger(name)
            logger.addFilter(log_filter)
            stack.callback(logger.removeFilter, log_filter)

        # NNAlgInterface.fit changes this unrelated logger as well as warning
        # filters. Do not let those upstream settings leak into user code.
        lightning_logger = logging.getLogger("lightning")
        stack.callback(lightning_logger.setLevel, lightning_logger.level)
        stack.callback(
            setattr, lightning_logger, "propagate", lightning_logger.propagate
        )
        stack.enter_context(warnings.catch_warnings())
        yield


class PytabkitCompatibilityMixin:
    def fit(self, *args, **kwargs):
        with backend_context():
            return super().fit(*args, **kwargs)

    def _predict_raw(self, *args, **kwargs):
        with backend_context():
            return super()._predict_raw(*args, **kwargs)


@contextmanager
def single_bin_diagnostics():
    # Two distinct values legitimately produce a single interval. The encoder
    # handles it correctly; the adapter records its count in training_info.
    # Keep bin validation and all other warnings active.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=(
                r"The \d+-th feature has just two bin edges, which means only one bin\."
                r" Strictly speaking, using a single bin for the piecewise-linear encoding"
                r" should not break anything, but it is the same as using"
                r" sklearn\.preprocessing\.MinMaxScaler\Z"
            ),
            category=UserWarning,
            module=r"pytabkit\.models\.nn_models\.rtdl_num_embeddings\Z",
        )
        yield
