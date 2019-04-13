import logging
import copy
import numpy as np
import pandas as pd
import time
import uuid
from supervised.tuner.registry import BINARY_CLASSIFICATION
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
)

log = logging.getLogger(__name__)


class ComputeAdditionalMetrics:
    @staticmethod
    def compute(target, predictions, ml_task):
        if ml_task != BINARY_CLASSIFICATION:
            return {}
        sorted_predictions = np.sort(predictions)
        STEPS = 100
        details = {
            "threshold": [],
            "f1": [],
            "accuracy": [],
            "precision": [],
            "recall": [],
            "mcc": [],
        }
        samples_per_step = max(1, np.floor(predictions.shape[0] / STEPS))

        for i in range(1, STEPS):
            idx = int(i * samples_per_step)
            if idx + 1 >= predictions.shape[0]:
                break
            th = 0.5 * (sorted_predictions[idx] + sorted_predictions[idx + 1])
            if np.sum(predictions > th) < 1:
                break
            response = (predictions > th).astype(int)

            details["threshold"] += [th]
            details["f1"] += [f1_score(target, response)]
            details["accuracy"] += [accuracy_score(target, response)]
            details["precision"] += [precision_score(target, response)]
            details["recall"] += [recall_score(target, response)]
            details["mcc"] += [matthews_corrcoef(target, response)]

        # max metrics
        max_metrics = {
            "auc": {
                "score": roc_auc_score(target, predictions),
                "threshold": None,
            },  # there is no threshold for AUC :)
            "f1": {
                "score": np.max(details["f1"]),
                "threshold": details["threshold"][np.argmax(details["f1"])],
            },
            "accuracy": {
                "score": np.max(details["accuracy"]),
                "threshold": details["threshold"][np.argmax(details["accuracy"])],
            },
            "precision": {
                "score": np.max(details["precision"]),
                "threshold": details["threshold"][np.argmax(details["precision"])],
            },
            "recall": {
                "score": np.max(details["recall"]),
                "threshold": details["threshold"][np.argmax(details["recall"])],
            },
            "mcc": {
                "score": np.max(details["mcc"]),
                "threshold": details["threshold"][np.argmax(details["mcc"])],
            },
        }
        # confusion matrix
        conf_matrix = confusion_matrix(
            target, predictions > max_metrics["f1"]["threshold"]
        )
        conf_matrix = pd.DataFrame(
            conf_matrix,
            columns=["Predicted as negative", "Predicted as positive"],
            index=["Labeled as negative", "Labeled as positive"],
        )

        return pd.DataFrame(details), pd.DataFrame(max_metrics), conf_matrix
