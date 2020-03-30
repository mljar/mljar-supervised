import logging
import copy
import numpy as np
import pandas as pd
import time
import uuid

from supervised.algorithms.registry import (
    BINARY_CLASSIFICATION,
    MULTICLASS_CLASSIFICATION,
    REGRESSION,
)
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from supervised.utils.metric import logloss

logger = logging.getLogger(__name__)
from supervised.utils.config import LOG_LEVEL

logger.setLevel(LOG_LEVEL)


class AdditionalMetrics:
    @staticmethod
    def binary_classification(target, predictions):

        predictions = np.array(predictions)
        sorted_predictions = np.sort(predictions)
        STEPS = 100  # can go lower for speed increase ???
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
            "logloss": {
                "score": logloss(target, predictions),
                "threshold": None,
            },  # there is no threshold for LogLoss
            "auc": {
                "score": roc_auc_score(target, predictions),
                "threshold": None,
            },  # there is no threshold for AUC
            "f1": {
                "score": np.max(details["f1"]),
                "threshold": details["threshold"][np.argmax(details["f1"])][0],
            },
            "accuracy": {
                "score": np.max(details["accuracy"]),
                "threshold": details["threshold"][np.argmax(details["accuracy"])][0],
            },
            "precision": {
                "score": np.max(details["precision"]),
                "threshold": details["threshold"][np.argmax(details["precision"])][0],
            },
            "recall": {
                "score": np.max(details["recall"]),
                "threshold": details["threshold"][np.argmax(details["recall"])][0],
            },
            "mcc": {
                "score": np.max(details["mcc"]),
                "threshold": details["threshold"][np.argmax(details["mcc"])][0],
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

        return {
            "metric_details": pd.DataFrame(details),
            "max_metrics": pd.DataFrame(max_metrics),
            "confusion_matrix": conf_matrix,
            "threshold": float(max_metrics["f1"]["threshold"]),
        }


    @staticmethod
    def multiclass_classification(target, predictions):

        all_labels = np.unique(target)
        # Print the confusion matrix
        conf_matrix = confusion_matrix(target, predictions, labels=all_labels)
        
        rows = [f"Predicted as {a}" for a in all_labels]
        cols = [f"Labeled as {a}" for a in all_labels]

        conf_matrix = pd.DataFrame(
                    conf_matrix,
                    columns=rows,
                    index=cols,
                )

        max_metrics = classification_report(target, predictions, digits=6, labels=all_labels, output_dict=True)

        print(pd.DataFrame(max_metrics).transpose())
        print(conf_matrix)

        return {
            "max_metrics": pd.DataFrame(max_metrics).transpose(),
            "confusion_matrix": conf_matrix,
        }
        
    @staticmethod
    def compute(target, predictions, ml_task):
        print(target, predictions, ml_task)
        if ml_task == BINARY_CLASSIFICATION:
            return AdditionalMetrics.binary_classification(target, predictions)
        elif ml_task == MULTICLASS_CLASSIFICATION:
            return AdditionalMetrics.multiclass_classification(target, predictions["label"])
        elif ml_task == REGRESSION:
            return None
