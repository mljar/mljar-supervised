import logging
import warnings
from numbers import Integral

import numpy as np
import sklearn
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.utils import _safe_indexing
from sklearn.utils.validation import check_consistent_length

from supervised.algorithms.registry import (
    BINARY_CLASSIFICATION,
    MULTICLASS_CLASSIFICATION,
    REGRESSION,
    AlgorithmsRegistry,
)
from supervised.algorithms.sklearn import SklearnAlgorithm
from supervised.utils.config import LOG_LEVEL

logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)


KNN_ROWS_LIMIT = 1000
KNN_INTERPRET_ROWS_LIMIT = 10000


class KNNFit(SklearnAlgorithm):
    def interpret(
        self,
        X_train,
        y_train,
        X_validation,
        y_validation,
        model_file_path,
        learner_name,
        target_name=None,
        class_names=None,
        metric_name=None,
        ml_task=None,
        explain_level=2,
    ):
        if explain_level <= 0:
            return
        # Use the full fold sizes, not the subsample used by fit(). See #97.
        train_rows = X_train.shape[0]
        validation_rows = X_validation.shape[0]
        if max(train_rows, validation_rows) > KNN_INTERPRET_ROWS_LIMIT:
            warnings.warn(
                f"Skipping kNN interpretation: training rows={train_rows}, "
                f"validation rows={validation_rows}; "
                f"row limit={KNN_INTERPRET_ROWS_LIMIT}.",
                stacklevel=2,
            )
            return
        # kNN supports permutation importance, but no SHAP explainer.
        super().interpret(
            X_train,
            y_train,
            X_validation,
            y_validation,
            model_file_path,
            learner_name,
            target_name=target_name,
            class_names=class_names,
            metric_name=metric_name,
            ml_task=ml_task,
            explain_level=1,
        )

    def file_extension(self):
        return "k_neighbors"

    def is_fitted(self):
        return (
            hasattr(self.model, "n_samples_fit_")
            and self.model.n_samples_fit_ is not None
            and self.model.n_samples_fit_ > 0
        )

    def fit(
        self,
        X,
        y,
        sample_weight=None,
        X_validation=None,
        y_validation=None,
        sample_weight_validation=None,
        log_to_file=None,
        max_time=None,
    ):
        rows_limit = self.params.get("rows_limit", KNN_ROWS_LIMIT)
        if (
            isinstance(rows_limit, bool)
            or not isinstance(rows_limit, Integral)
            or rows_limit < 1
        ):
            raise ValueError("Nearest Neighbors rows_limit must be a positive integer")
        check_consistent_length(X, y)
        selected_rows = min(X.shape[0], rows_limit)
        neighbors = self.model.n_neighbors
        if (
            isinstance(neighbors, bool)
            or not isinstance(neighbors, Integral)
            or neighbors < 1
        ):
            raise ValueError("Nearest Neighbors n_neighbors must be a positive integer")
        if neighbors > selected_rows:
            raise ValueError(
                f"Nearest Neighbors n_neighbors={neighbors} exceeds the {selected_rows} "
                "available training rows. Decrease n_neighbors, increase rows_limit, "
                "or provide more training rows."
            )
        if X.shape[0] > rows_limit:
            if isinstance(self.model, KNeighborsRegressor):
                indices, _ = train_test_split(
                    np.arange(X.shape[0]), train_size=rows_limit, random_state=1234
                )
            else:
                indices = self._classification_sample(y, rows_limit)
            self.model.fit(_safe_indexing(X, indices), _safe_indexing(y, indices))
        else:
            self.model.fit(X, y)

    @staticmethod
    def _classification_sample(y, rows_limit):
        labels = np.asarray(y)
        if labels.ndim == 2 and labels.shape[1] == 1:
            labels = labels[:, 0]
        if labels.ndim != 1:
            raise ValueError(
                "Nearest Neighbors classification requires a one-dimensional target"
            )
        classes, inverse, counts = np.unique(
            labels, return_inverse=True, return_counts=True
        )
        class_count = len(classes)
        if class_count > rows_limit:
            raise ValueError(
                f"Nearest Neighbors rows_limit={rows_limit} cannot retain all {class_count} "
                "training classes. Increase rows_limit to at least the number of classes "
                "or choose another algorithm."
            )
        if counts.min() < 2:
            reason = "a training class has only one sample"
        elif len(labels) - rows_limit < class_count:
            reason = "the discarded partition is too small for stratification"
        else:
            indices, _ = train_test_split(
                np.arange(len(labels)),
                train_size=rows_limit,
                stratify=y,
                random_state=1234,
            )
            if len(np.unique(labels[indices])) == class_count:
                return indices
            reason = "the stratified sample omitted a training class"

        # Reserve one row per class. Allocate the remaining slots proportionally
        # to the remaining class populations, with seeded tie breaking. Select
        # positions without replacement; no validation rows enter this sample.
        rng = np.random.RandomState(1234)
        allocation = np.ones(class_count, dtype=np.int64)
        remaining = rows_limit - class_count
        if remaining:
            quotas = (
                remaining * (counts - 1).astype(float) / (len(labels) - class_count)
            )
            extra = np.floor(quotas).astype(np.int64)
            allocation += extra
            leftover = remaining - int(extra.sum())
            order = rng.permutation(class_count)
            order = order[np.argsort(-(quotas - extra)[order], kind="stable")]
            allocation[order[:leftover]] += 1
        groups = np.split(np.argsort(inverse, kind="stable"), np.cumsum(counts)[:-1])
        indices = rng.permutation(
            np.concatenate(
                [
                    rng.choice(group, size=int(size), replace=False)
                    for group, size in zip(groups, allocation)
                ]
            )
        )
        warnings.warn(
            f"Nearest Neighbors used class-preserving subsampling: {reason}; "
            f"selected {len(indices)} of {len(labels)} rows across {class_count} classes.",
            UserWarning,
            stacklevel=3,
        )
        return indices

    @property
    def _classes(self):
        # Returns the unique classes based on the fitted model
        if hasattr(self.model, "classes_"):
            return self.model.classes_
        else:
            return None


class KNeighborsAlgorithm(ClassifierMixin, KNNFit):
    algorithm_name = "k-Nearest Neighbors"
    algorithm_short_name = "Nearest Neighbors"

    def __init__(self, params):
        super(KNeighborsAlgorithm, self).__init__(params)
        logger.debug("KNeighborsAlgorithm.__init__")
        self.library_version = sklearn.__version__
        self.max_iters = 1
        self.model = KNeighborsClassifier(
            n_neighbors=params.get("n_neighbors", 3),
            weights=params.get("weights", "uniform"),
            algorithm="kd_tree",
            n_jobs=params.get("n_jobs", -1),
        )


class KNeighborsRegressorAlgorithm(RegressorMixin, KNNFit):
    algorithm_name = "k-Nearest Neighbors"
    algorithm_short_name = "Nearest Neighbors"

    def __init__(self, params):
        super(KNeighborsRegressorAlgorithm, self).__init__(params)
        logger.debug("KNeighborsRegressorAlgorithm.__init__")
        self.library_version = sklearn.__version__
        self.max_iters = 1
        self.model = KNeighborsRegressor(
            n_neighbors=params.get("n_neighbors", 3),
            weights=params.get("weights", "uniform"),
            algorithm="ball_tree",
            n_jobs=params.get("n_jobs", -1),
        )


knn_params = {"n_neighbors": [3, 5, 7], "weights": ["uniform", "distance"]}

default_params = {"n_neighbors": 5, "weights": "uniform"}

additional = {"max_rows_limit": 100000, "max_cols_limit": 100}

required_preprocessing = [
    "missing_values_inputation",
    "convert_categorical",
    "datetime_transform",
    "text_transform",
    "ip_transform",
    "scale",
    "target_as_integer",
]

AlgorithmsRegistry.add(
    BINARY_CLASSIFICATION,
    KNeighborsAlgorithm,
    knn_params,
    required_preprocessing,
    additional,
    default_params,
)
AlgorithmsRegistry.add(
    MULTICLASS_CLASSIFICATION,
    KNeighborsAlgorithm,
    knn_params,
    required_preprocessing,
    additional,
    default_params,
)

AlgorithmsRegistry.add(
    REGRESSION,
    KNeighborsRegressorAlgorithm,
    knn_params,
    required_preprocessing,
    additional,
    default_params,
)
