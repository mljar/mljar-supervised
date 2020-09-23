import os
import json
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from supervised.algorithms.registry import (
    BINARY_CLASSIFICATION,
    MULTICLASS_CLASSIFICATION,
    REGRESSION,
)
from supervised.utils.subsample import subsample

logger = logging.getLogger(__name__)
from supervised.utils.config import LOG_LEVEL

logger.setLevel(LOG_LEVEL)

from sklearn.metrics import make_scorer, log_loss
import sys


def log_loss_eps(y_true, y_pred):
    ll = log_loss(y_true, y_pred, eps=1e-7)
    return ll


log_loss_scorer = make_scorer(log_loss_eps, greater_is_better=False, needs_proba=True)


class PermutationImportance:
    @staticmethod
    def compute_and_plot(
        model,
        X_validation,
        y_validation,
        model_file_path,
        learner_name,
        metric_name=None,
        ml_task=None,
    ):
        # for scoring check https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
        if ml_task == BINARY_CLASSIFICATION:
            scoring = log_loss_scorer
        elif ml_task == MULTICLASS_CLASSIFICATION:
            scoring = log_loss_scorer
        else:
            scoring = "neg_mean_squared_error"

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # subsample validation data to speed-up importance computation
                # in the case of large number of columns, it can take a lot of time
                rows, cols = X_validation.shape
                if cols > 5000:
                    X_vald, _, y_vald, _ = subsample(
                        X_validation, y_validation, train_size=100, ml_task=ml_task
                    )
                elif cols > 50 and rows * cols > 200000:
                    X_vald, _, y_vald, _ = subsample(
                        X_validation, y_validation, train_size=1000, ml_task=ml_task
                    )
                else:
                    X_vald = X_validation
                    y_vald = y_validation

                importance = permutation_importance(
                    model,
                    X_vald,
                    y_vald,
                    scoring=scoring,
                    n_jobs=-1,  # all cores
                    random_state=12,
                    n_repeats=5,  # default
                )

            sorted_idx = importance["importances_mean"].argsort()

            # save detailed importance
            df_imp = pd.DataFrame(
                {
                    "feature": X_vald.columns[sorted_idx],
                    "mean_importance": importance["importances_mean"][sorted_idx],
                }
            )
            df_imp.to_csv(
                os.path.join(model_file_path, f"{learner_name}_importance.csv"),
                index=False,
            )
        except Exception as e:
            print("Problem during computing permutation importance. Skipping ...")
