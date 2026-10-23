"""Reuse completed-fold costs without treating every architecture as equivalent."""

import copy
import math

import joblib

from supervised.tuner.algorithm_early_stopping import AlgorithmEarlyStopping


class StackingCostHistory:
    def __init__(self, observations=None):
        self.observations = copy.deepcopy(observations or [])

    @staticmethod
    def descriptor(params, inputs):
        learner = dict(params["learner"], **params.get("custom_params", {}))
        algorithm = learner.get("model_type")
        for key in ("seed", "random_state", "model_architecture_json", "explain_level"):
            learner.pop(key, None)
        architecture = None
        if algorithm == "RealMLP":
            architecture = [learner.pop("n_hidden_layers", 2), learner.pop("hidden_width", 64)]
            learner["dropout_enabled"] = bool(learner.get("p_drop", 0.15))
            learner["weight_decay_enabled"] = bool(learner.get("wd", 0.02))
            # These change optimization, but not the network's operation sizes.
            for key in ("lr", "p_drop", "wd"):
                learner.pop(key, None)
        return dict(
            algorithm=algorithm, architecture=architecture,
            signature=joblib.hash((learner, inputs, params.get("preprocessing", {}),
                                   AlgorithmEarlyStopping.validation_key(params),
                                   params.get("explain_level"), params.get("data_type"))),
        )

    def observe(self, name, descriptor, elapsed, completed_folds, expected_folds, dimensions=None):
        if not descriptor or not 0 < completed_folds <= expected_folds:
            return
        estimate = elapsed * expected_folds / completed_folds
        if not math.isfinite(estimate) or estimate <= 0:
            return
        observation = dict(name=name, descriptor=copy.deepcopy(descriptor),
                           elapsed=elapsed, completed_folds=completed_folds,
                           expected_folds=expected_folds, estimated_total=estimate,
                           dimensions=copy.deepcopy(dimensions))
        self.observations = [o for o in self.observations if o["name"] != name]
        self.observations.append(observation)

    def decision(self, descriptor, allowance):
        if allowance is None or not descriptor:
            return None
        comparable = []
        for observation in self.observations:
            previous = observation["descriptor"]
            if previous["signature"] != descriptor["signature"]:
                continue
            shape, old_shape = descriptor["architecture"], previous["architecture"]
            # A smaller network gets a probe. Mixed width/depth changes are
            # incomparable; never assume linear runtime scaling with size.
            if shape is not None and not all(a >= b for a, b in zip(shape, old_shape)):
                continue
            comparable.append(observation)
        if not comparable:
            return None
        # Prefer the cheapest measured compatible estimate, allowing a faster
        # subsequent observation to supersede an earlier pessimistic probe.
        source = min(comparable, key=lambda o: o["estimated_total"])
        if source["estimated_total"] <= allowance:
            return None
        return dict(phase="stacking_admission", source_attempt=source["name"],
                    estimated_total=source["estimated_total"], allowance=allowance,
                    comparison="equivalent or no-smaller architecture with compatible inputs")
