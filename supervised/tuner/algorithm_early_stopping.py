"""Stop uncompetitive original-feature searches using validation history only."""

import copy
import hashlib
import json
import math
import os

from supervised.utils.jsonencoder import MLJSONEncoder
from supervised.utils.metric import Metric


class AlgorithmEarlyStopping:
    VERSION = 1
    MIN_CONFIGURATIONS = 3
    MAX_GAP = 0.20

    def __init__(self, metric, enabled=True):
        self.metric = metric
        self.enabled = enabled
        self.minimize = Metric({"name": metric}).get_minimize_direction()
        self.validation = None
        self.reference = None
        self.leader = None
        self.algorithms = {}
        self.paused = {}
        self.skipped = {}
        self.reported = []

    @staticmethod
    def tuning_step(step):
        return step == "not_so_random" or step.startswith("hill_climbing_")

    @staticmethod
    def original(params):
        preprocessing = params.get("preprocessing", {})
        return (
            params.get("data_type", "original") == "original"
            and not params.get("is_stacked", False)
            and "RandomFeature" not in params.get("name", "")
            and params.get("learner", {}).get("model_type") not in (None, "Ensemble")
            and not any(preprocessing.get(k) for k in (
                "golden_features", "feature_discovery", "kmeans_features", "drop_features"
            ))
        )

    @staticmethod
    def validation_key(params):
        validation = dict(params.get("validation_strategy", {}))
        # A moved results directory must not change a resumed decision. Preserve
        # split/weight file identities, but not the original absolute directory.
        for key in list(validation):
            if key in ("X_path", "y_path", "results_path"):
                del validation[key]
            elif key.endswith("_path") and isinstance(validation[key], str):
                validation[key] = os.path.basename(validation[key])
        return json.dumps(validation, sort_keys=True, cls=MLJSONEncoder)

    @staticmethod
    def configuration_key(params):
        effective = dict(params["learner"], **params.get("custom_params", {}))
        learner = {k: v for k, v in effective.items() if k not in (
            "seed", "random_state", "n_jobs", "explain_level", "model_architecture_json"
        )}
        payload = json.dumps([learner, params.get("preprocessing", {})],
                             sort_keys=True, cls=MLJSONEncoder)
        return hashlib.sha256(payload.encode()).hexdigest()

    def observe(self, step, params, final_loss):
        observation_step = step in ("simple_algorithms", "default_algorithms", "not_so_random") or step.startswith("hill_climbing_")
        if (not self.enabled or not observation_step
                or params.get("status") != "trained" or not self.original(params)):
            return
        try:
            loss = float(final_loss) * (1 if self.minimize else -1)
        except (ValueError, TypeError):
            return
        if not math.isfinite(loss):
            return
        validation = self.validation_key(params)
        if self.validation is not None and validation != self.validation:
            return
        self.validation = validation
        record = {"model": params["name"], "loss": loss}
        if self.reference is None:
            self.reference = record
        if self.leader is None or loss < self.leader["loss"]:
            self.leader = record
        algorithm = params["learner"]["model_type"]
        state = self.algorithms.setdefault(algorithm, {"configurations": [], "best": record})
        key = self.configuration_key(params)
        if key not in state["configurations"]:
            state["configurations"].append(key)
        if loss < state["best"]["loss"]:
            state["best"] = record

    def decision(self, step, params):
        if (not self.enabled or not self.tuning_step(step) or not self.original(params)
                or self.validation_key(params) != self.validation):
            return None
        algorithm = params["learner"]["model_type"]
        state = self.algorithms.get(algorithm)
        if not state or len(state["configurations"]) < self.MIN_CONFIGURATIONS:
            return None
        gain = self.reference["loss"] - self.leader["loss"]
        gap = state["best"]["loss"] - self.leader["loss"]
        # Differences, not absolute scores, make the rule insensitive to units
        # and score offsets. Avoid dividing by zero or overflowing differences.
        if not math.isfinite(gain) or not math.isfinite(gap) or gain <= 0:
            return None
        boundary = self.MAX_GAP * gain
        if gap <= boundary or math.isclose(gap, boundary, rel_tol=1e-12, abs_tol=0):
            return None
        if not math.isfinite(gap / gain):
            return None
        return copy.deepcopy(dict(
            algorithm=algorithm, successful_configurations=len(state["configurations"]),
            reference=self.reference, leader=self.leader, algorithm_best=state["best"],
            normalized_gap=gap / gain, threshold=self.MAX_GAP,
            reason="less than 80% of the leader's improvement over the fixed reference",
        ))

    def skip_candidate(self, step, params):
        decision = self.decision(step, params)
        if decision is None:
            return False
        params.update(status="skipped", final_loss=None, train_time=None,
                      algorithm_early_stopping=decision)
        self.paused.setdefault(decision["algorithm"], decision)
        self.skipped[f"{step}/{params['name']}"] = dict(
            step=step, name=params["name"], learner=copy.deepcopy(params["learner"]),
            **decision,
        )
        return True

    def filter_candidates(self, step, candidates):
        # Retain skipped metadata for progress/resume, without allocating models.
        for params in candidates:
            if params.get("status") not in ("trained", "skipped", "error"):
                self.skip_candidate(step, params)
        return candidates

    def messages(self):
        messages = []
        for algorithm, decision in self.paused.items():
            if algorithm not in self.reported:
                messages.append(
                    f"Algorithm early stopping: {algorithm} paused original-feature tuning "
                    f"after {decision['successful_configurations']} configurations "
                    f"(normalized gap {decision['normalized_gap']:.1%}, limit 20%)."
                )
                self.reported.append(algorithm)
        return messages

    def to_json(self):
        return copy.deepcopy(dict(
            version=self.VERSION, metric=self.metric, enabled=self.enabled,
            threshold=self.MAX_GAP, min_configurations=self.MIN_CONFIGURATIONS,
            validation=self.validation, reference=self.reference, leader=self.leader,
            algorithms=self.algorithms, paused=self.paused, skipped=self.skipped,
            reported=self.reported,
        ))

    @classmethod
    def from_json(cls, data):
        if data["version"] != cls.VERSION:
            raise ValueError("Unsupported algorithm early stopping policy version")
        policy = cls(data["metric"], enabled=data["enabled"])
        for key in ("validation", "reference", "leader", "algorithms", "paused", "skipped", "reported"):
            setattr(policy, key, copy.deepcopy(data[key]))
        return policy
