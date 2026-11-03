"""Keep fold-level scoring separate from averaged out-of-fold predictions."""

import numpy as np
import pandas as pd


FOLD_INDEX = ["validation_fold", "validation_row"]


def collect_fold_predictions(frames):
    return pd.concat(frames, keys=range(len(frames)), names=FOLD_INDEX).sort_index()


def has_fold_index(frame):
    return isinstance(frame.index, pd.MultiIndex) and list(frame.index.names) == FOLD_INDEX


def mean_fold_score(metric, target, predictions, sample_weight=None):
    """Give every completed fold equal weight, including across CV repeats."""
    if not has_fold_index(predictions):
        # Legacy OOF artifacts and direct, single-validation-set callers.
        return metric(target, predictions, sample_weight)
    target = np.asarray(target)
    weights = None if sample_weight is None else np.asarray(sample_weight)
    values = np.asarray(predictions)
    scores = [
        metric(target[rows], values[rows], None if weights is None else weights[rows])
        for rows in predictions.groupby(level="validation_fold", sort=False).indices.values()
    ]
    return float(np.mean(scores))


def average_fold_predictions(frame):
    """One prediction per row for stacking, reports and downstream features."""
    if not has_fold_index(frame):
        return frame
    result = frame.groupby(level="validation_row", sort=True).agg(
        {col: "mean" if "prediction" in col else "first" for col in frame.columns}
    )
    result.index.name = None
    return result
