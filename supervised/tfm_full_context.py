"""One full-data TFM for inference, with untouched CV artifacts for validation."""
import copy
import time

from supervised.algorithms.factory import AlgorithmFactory
from supervised.exceptions import AutoMLException, BudgetExceededError
from supervised.preprocessing.preprocessing import Preprocessing
from supervised.utils.artifact_paths import portable_artifact_path, resolve_artifact_path
from supervised.utils.utils import data_row_count, load_data


class TFMFullContext:
    def __init__(self, framework, results_path, model_subpath, budget_started):
        self.framework = framework
        self.root = results_path
        self.subpath = model_subpath
        self.started = time.monotonic()
        self.deadlines = []
        limit = framework.params.get("candidate_time_limit")
        if limit is not None:
            self.deadlines.append(budget_started + limit)
        total = framework.callbacks.get("total_time_constraint")
        if total is not None and total.total_time_limit is not None:
            self.deadlines.append(time.monotonic() + total.total_time_limit
                                  - (time.time() - total.total_time_start))
        self.prepare()

    def remaining(self):
        remaining = self.framework._max_time_for_learner
        if self.deadlines:
            remaining = min(remaining, min(self.deadlines) - time.monotonic())
        if remaining <= 0:
            raise BudgetExceededError("TFM full-context preparation exceeded the training budget.",
                                      phase="tfm_full_context")
        return remaining

    def prepare(self):
        model = self.framework
        params = copy.deepcopy(model.learner_params)
        params.update(model.params.get("custom_params", {}))
        params["explain_level"] = model._explain_level
        learner = AlgorithmFactory.get_algorithm(params)
        learner.name = "learner_full"
        validation = model.validation_params
        # Check the full count before copying data or running any CV learners.
        self.rows = data_row_count(validation["X_path"], store=model._data_store)
        learner._check_training_size(self.rows)
        self.remaining()
        preprocessing = copy.deepcopy(model.preprocessing_params)
        from supervised.preprocessing.feature_identity import reject_generated_drops
        reject_generated_drops(preprocessing)
        if preprocessing.get("feature_selection_manifest") is not None:
            raise AutoMLException("A fitted feature selection manifest cannot be transplanted into a full-data TFM generator.")
        if "feature_discovery" in preprocessing:
            from supervised.preprocessing.feature_discovery_transformer import FeatureDiscovery

            discovery = FeatureDiscovery(self.root, model._ml_task,
                                         model._automl_random_state, params.get("n_jobs", -1),
                                         data_store=model._data_store)
            prepared = discovery.prepare(validation, self.remaining(), full_data=True)
            profile = preprocessing["feature_discovery"]["feature_set"]
            if prepared is None or profile not in prepared["folds"]["0:0"]["sets"]:
                raise AutoMLException("Feature discovery profile is unavailable for the full TFM context.")
            preprocessing["feature_discovery"]["folds"] = {"0:full": prepared["folds"]["0:0"]}
        prep = Preprocessing(preprocessing, model.get_name(), "full", 0,
                             results_path=self.root)
        X = load_data(validation["X_path"], store=model._data_store)
        y = load_data(validation["y_path"], store=model._data_store)["target"]
        weights = None
        if validation.get("sample_weight_path"):
            weights = load_data(validation["sample_weight_path"], store=model._data_store)["sample_weight"]
        X, y, weights = prep.fit_and_transform(X, y, weights)
        learner.fit_full_context(
            X, y, sample_weight=weights,
            log_to_file=resolve_artifact_path(
                self.root, portable_artifact_path(self.subpath, "learner_full_training.log")),
            max_time=self.remaining(),
        )
        self.remaining()
        self.saved = portable_artifact_path(self.subpath, learner.get_fname())
        path = resolve_artifact_path(self.root, self.saved)
        learner.save(path, weights_root=self.root)
        self.remaining()
        description = learner.get_params()
        learner.unload()
        # Restore only metadata: do not retain full context alongside CV folds.
        self.learner = AlgorithmFactory.load(description, path, lazy_load=True)
        self.preprocessing = prep
        self.prepare_seconds = time.monotonic() - self.started

    def activate(self):
        self.remaining()
        model = self.framework
        # CV learners remain on disk for inspection. Prediction/export sees only
        # learner_full. Older frameworks without this metadata still average folds.
        model._validation_models = {
            "learners": [learner.get_params() for learner in model.learners],
            "preprocessing": [prep.to_json() for prep in model.preprocessings],
            "saved": [portable_artifact_path(self.subpath, learner.get_fname())
                      for learner in model.learners],
        }
        model._tfm_full_context = {
            "context_rows": self.rows,
            "validation_learners": len(model.learners),
            "prepare_seconds": self.prepare_seconds,
            "selection": "cross_validation",
        }
        model.learners = [self.learner]
        model.preprocessings = [self.preprocessing]
