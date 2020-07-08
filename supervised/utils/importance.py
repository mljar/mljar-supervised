import os
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

logger = logging.getLogger(__name__)
from supervised.utils.config import LOG_LEVEL

logger.setLevel(LOG_LEVEL)


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
            scoring = "neg_log_loss"
        elif ml_task == MULTICLASS_CLASSIFICATION:
            scoring = "neg_log_loss"
        else:
            scoring = "neg_mean_squared_error"

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            importance = permutation_importance(
                model,
                X_validation,
                y_validation,
                scoring=scoring,
                n_jobs=-1,
                random_state=12,
            )

        sorted_idx = importance["importances_mean"].argsort()

        # save detailed importance
        df_imp = pd.DataFrame(
            {
                "feature": X_validation.columns[sorted_idx],
                "mean_importance": importance["importances_mean"][sorted_idx],
            }
        )
        df_imp.to_csv(
            os.path.join(model_file_path, f"{learner_name}_importance.csv"), index=False
        )

        """
        # Do not plot. We will plot aggregate of all importances.
        # limit number of column in the plot
        if len(sorted_idx) > 50:
            sorted_idx = sorted_idx[-50:]
        plt.figure(figsize=(10, 7))
        plt.barh(
            X_validation.columns[sorted_idx], importance["importances_mean"][sorted_idx]
        )
        plt.xlabel("Mean of feature importance")
        plt.tight_layout(pad=2.0)
        plot_path = os.path.join(model_file_path, f"{learner_name}_importance.png")
        plt.savefig(plot_path)
        plt.close("all")
        """
