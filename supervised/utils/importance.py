import logging
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from supervised.algorithms.registry import (
    BINARY_CLASSIFICATION,
    MULTICLASS_CLASSIFICATION,
)

logger = logging.getLogger(__name__)
from supervised.utils.config import LOG_LEVEL

logger.setLevel(LOG_LEVEL)

from sklearn.metrics import log_loss, make_scorer


def log_loss_eps(y_true, y_pred, labels=None):
    ll = log_loss(y_true, y_pred, labels=labels)
    return ll


log_loss_scorer = make_scorer(
    log_loss_eps, greater_is_better=False, response_method="predict_proba"
)


class PermutationImportance:
    @staticmethod
    def compute(model, X_validation, y_validation, ml_task=None, n_jobs=-1):
        """Return permutation importance in the learner's feature/target space."""
        scoring = (
            log_loss_scorer
            if ml_task in (BINARY_CLASSIFICATION, MULTICLASS_CLASSIFICATION)
            else "neg_mean_squared_error"
        )
        if scoring is log_loss_scorer and hasattr(model, "classes_"):
            # Evaluation data need not contain every class seen during fit.
            scoring = make_scorer(
                log_loss_eps,
                greater_is_better=False,
                response_method="predict_proba",
                labels=model.classes_,
            )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rows, cols = X_validation.shape
            sample_size = None
            if cols > 5000 and rows > 100:
                sample_size = 100
            elif cols > 50 and rows * cols > 200000 and rows > 1000:
                sample_size = 1000
            X_vald, y_vald = X_validation, y_validation
            if sample_size is not None:
                X_vald, _, y_vald, _ = train_test_split(
                    X_validation,
                    y_validation,
                    train_size=sample_size,
                    stratify=(
                        y_validation
                        if ml_task in (BINARY_CLASSIFICATION, MULTICLASS_CLASSIFICATION)
                        else None
                    ),
                    random_state=12,
                )
            importance = permutation_importance(
                model,
                X_vald,
                y_vald,
                scoring=scoring,
                n_jobs=n_jobs,
                random_state=12,
                n_repeats=5,
            )
        return (
            pd.DataFrame(
                {
                    "feature": X_vald.columns,
                    "mean_importance": importance["importances_mean"],
                }
            )
            .sort_values("mean_importance", kind="stable")
            .reset_index(drop=True)
        )

    @staticmethod
    def compute_and_plot(
        model,
        X_validation,
        y_validation,
        model_file_path,
        learner_name,
        metric_name=None,
        ml_task=None,
        n_jobs=-1,
    ):
        # Keep the training hook and its best-effort behavior backwards compatible.
        try:
            df_imp = PermutationImportance.compute(
                model, X_validation, y_validation, ml_task, n_jobs
            )
            df_imp.to_csv(
                os.path.join(model_file_path, f"{learner_name}_importance.csv"),
                index=False,
            )
        except Exception as e:
            print(str(e))
            print("Problem during computing permutation importance. Skipping ...")


def load_importance_file(path):
    """Read one permutation CSV; missing or invalid artifacts are unavailable."""
    try:
        frame = pd.read_csv(path, dtype={"feature": str}, keep_default_na=False)
        if frame.empty or not {"feature", "mean_importance"}.issubset(frame.columns):
            return None
        if frame["feature"].duplicated().any() or (frame["feature"] == "").any():
            return None
        values = pd.to_numeric(frame["mean_importance"], errors="raise").astype(float)
        if not np.isfinite(values).all():
            return None
        return pd.Series(values.to_numpy(), index=frame["feature"], name="importance")
    except (OSError, ValueError, pd.errors.ParserError):
        return None


def aggregate_importance(vectors):
    """Average available learners, aligning features and skipping absent features."""
    vectors = [v for v in vectors if v is not None and not v.empty]
    if not vectors:
        return None
    return pd.concat(vectors, axis=1).mean(axis=1).rename("importance")


def load_model_importance(model_dir):
    """Load and average saved permutation importance (excluding SHAP artifacts)."""
    try:
        paths = sorted(
            os.path.join(model_dir, name)
            for name in os.listdir(model_dir)
            if name.endswith("_importance.csv") and "shap" not in name
        )
    except OSError:
        return None
    return aggregate_importance(load_importance_file(path) for path in paths)
