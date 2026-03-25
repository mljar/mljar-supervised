__version__ = "1.2.0"

import numpy as np


def _ensure_numpy_compat():
    if not hasattr(np, "in1d"):
        # NumPy 2.4 removed np.in1d; some dependencies still use it.
        np.in1d = np.isin


_ensure_numpy_compat()

from supervised.automl import AutoML
