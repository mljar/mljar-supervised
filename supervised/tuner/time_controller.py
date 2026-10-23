import logging
import time

import numpy as np

from supervised.utils.config import LOG_LEVEL
from supervised.tuner.stacking_cost import StackingCostHistory

logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)


class TimeController:
    def __init__(
        self, start_time, total_time_limit, model_time_limit, steps, algorithms
    ):
        self._start_time = start_time
        self._total_time_limit = total_time_limit
        self._model_time_limit = model_time_limit
        self._steps = steps
        self._algorithms = algorithms
        self._spend = []
        self._is_hill_climbing = "hill_climbing_1" in steps
        self._is_stacking = "stack" in steps
        self._active_algorithms = None
        self._active_step = None
        self._stage_budgets = {}
        self.stacking_cost = StackingCostHistory()

    def set_active_algorithms(self, step, algorithms):
        # Reconstructed from pending candidates on resume, never a permanent
        # blacklist: a changed feature representation remains eligible.
        self._active_step = step
        self._active_algorithms = set(algorithms)

    def candidate_budget(self, model_type, step, configuration):
        """Whole-trial allowance; unsuccessful work also consumes exploration."""
        if self._model_time_limit is not None:
            return self._model_time_limit
        if self._total_time_limit is None:
            return None
        remaining = self.remaining_step_time(step)
        families = (self._active_algorithms if self._active_step == step else None) or set(self._algorithms)
        pool = self._stage_budgets.setdefault(step, remaining + self.step_spend(step))
        spent = self.model_spend(model_type, step)
        allowance = max(0.0, pool / max(1, len(families)) - spent)
        budget = min(remaining, allowance)
        # A timeout is only a lower bound. Retry this exact representation and
        # configuration only with a larger budget, never a smaller one.
        bounds = [s["train_time"] for s in self._spend
                  if s.get("configuration") == configuration
                  and s.get("outcome") == "budget_exhausted"]
        if bounds and budget <= max(bounds):
            return 0.0
        return budget

    def record_attempt(self, model_name, model_type, step, elapsed, outcome, configuration):
        # keep_model logs successful fits; replace that entry with complete
        # attempt wall time (including preparation, prediction and saving).
        self._spend = [s for s in self._spend
                       if not (s["model_name"] == model_name and s["fit_level"] == step)]
        self._spend.append(dict(model_name=model_name, model_type=model_type,
                                fit_level=step, train_time=elapsed, outcome=outcome,
                                configuration=configuration))

    def to_json(self):
        return {
            "total_time_limit": self._total_time_limit,
            "model_time_limit": self._model_time_limit,
            "steps": self._steps,
            "algorithms": self._algorithms,
            "spend": self._spend,
            "is_hill_climbing": self._is_hill_climbing,
            "is_stacking": self._is_stacking,
            "stage_budgets": self._stage_budgets,
            "stacking_cost_observations": self.stacking_cost.observations,
        }

    @staticmethod
    def from_json(data):
        if data is None:
            return None
        try:
            total_time_limit = data.get("total_time_limit")
            model_time_limit = data.get("model_time_limit")
            steps = data.get("steps")
            algorithms = data.get("algorithms")

            tc = TimeController(
                time.time(), total_time_limit, model_time_limit, steps, algorithms
            )
            tc._spend = data.get("spend")
            tc._stage_budgets = data.get("stage_budgets", {})
            tc.stacking_cost = StackingCostHistory(data.get("stacking_cost_observations"))
            tc._start_time -= tc.already_spend()  # update time with already spend
            return tc
        except Exception as e:
            logger.error(f"Cant load TimeController from json, {str(e)}")
            pass
        return None

    def already_spend(self):
        return np.sum([s["train_time"] for s in self._spend])

    def time_should_use(self, fit_level):
        if self._total_time_limit is None:
            return 7 * 24 * 3600  # 7 days

        ratios = {
            "default_algorithms": 0.3,
            "not_so_random": 0.35,
            "mix_encoding": 0.05,
            "golden_features": 0.05,
            "kmeans_features": 0.05,
            "insert_random_feature": 0.05,
            "features_selection": 0.05,
            "hill_climbing_1": 0.2,  # enough to have only first step from hill climbing
            "boost_on_errors": 0.05,
            "stack": 0.2,
        }
        if "feature_discovery" in self._steps:
            ratios["feature_discovery"] = 0.025 if "golden_features" in self._steps else 0.05
            if "golden_features" in self._steps:
                ratios["golden_features"] = 0.025
            if "mix_encoding" in self._steps:
                ratios["mix_encoding"] = 0.025
                ratios["feature_discovery"] += 0.025

        # Preserve the enabled feature stages' combined budget, but distribute
        # it in 30/50/20 proportions. Disabled generators receive no allocation.
        feature_shares = {
            "golden_features": 0.3,
            "feature_discovery": 0.5,
            "kmeans_features": 0.2,
        }
        enabled_features = [step for step in feature_shares if step in self._steps]
        feature_budget = sum(ratios[step] for step in enabled_features)
        enabled_share = sum(feature_shares[step] for step in enabled_features)
        for step in enabled_features:
            ratios[step] = feature_budget * feature_shares[step] / enabled_share

        if (
            fit_level
            in [
                "default_algorithms",
                "not_so_random",
                "boost_on_errors",
                "mix_encoding",
                "golden_features",
                "feature_discovery",
                "kmeans_features",
                "insert_random_feature",
                "features_selection",
                "stack",
            ]
            or "hill_climbing" in fit_level
        ):
            ratio = 0
            for k, v in ratios.items():
                if k in self._steps:
                    ratio += v

            fl = fit_level
            if "hill_climbing" in fit_level:
                fl = "hill_climbing_1"

            ratio = ratios[fl] / ratio

            if "hill_climbing" in fit_level:
                # print("before hill climbing scale", ratio)
                hill_climbing_cnt = len(
                    [i for i in self._steps if "hill_climbing" in i]
                )
                ratio /= float(hill_climbing_cnt)

            should_use = self._total_time_limit * ratio

            return should_use

        return 0

    def compound_time_should_use(self, fit_level):
        compound = 0
        for step in self._steps:
            if step in [
                "adjust_validation",
                "simple_algorithms",
                # "default_algorithms",
                "ensemble",
                "ensemble_stacked",
            ]:
                continue
            time_should_use = self.time_should_use(step)
            compound += time_should_use

            if fit_level == step:
                break
        # if fit_level == "stack":
        #    compound -= 120 # leave time for ensemble
        # maybe not needed
        return compound

    def remaining_step_time(self, fit_level):
        """Wall-clock allowance, including unused time from earlier steps."""
        if self._total_time_limit is None or self._model_time_limit is not None:
            return float("inf")
        elapsed = time.time() - self._start_time
        return max(0.0, min(self._total_time_limit, self.compound_time_should_use(fit_level)) - elapsed)

    def enough_time_for_step(self, fit_level):
        if fit_level in ["ensemble", "ensemble_stacked", "fairness"]:
            return True
        total_time_spend = time.time() - self._start_time
        compound = self.compound_time_should_use(fit_level)
        # print("Enough time for step", fit_level, np.round(total_time_spend,2), np.round(compound,2))
        if total_time_spend > compound:
            # dont train more
            return False

        return True

    def enough_time_for_model(self, model_type, step=None, expected_time=None):
        if self._total_time_limit is None:
            return True

        time_left = self._total_time_limit - (time.time() - self._start_time)
        if step is not None:
            time_left = min(time_left, self.remaining_step_time(step))
        if expected_time is None:
            spend = [s["train_time"] for s in self._spend if s["model_type"] == model_type]
            expected_time = np.mean(spend)
        return expected_time <= time_left

    def enough_time(self, model_type, step, expected_time=None):
        """
        Check if there is enough time to train the next model.

        Parameters
        ----------
        model_type : str
            String with type of the model.

        step: str
            String with name of the step in the process of AutoML training.

        expected_time: float or None
            Candidate-specific training estimate in seconds. Otherwise use the
            mean training time of earlier models of this algorithm.

        Returns
        -------
        bool
            `True` if there is time for training next model, `False` otherwise.
        """
        if step in ["ensemble", "ensemble_stacked"]:
            return True
        # if model_time_limit is set, train every model
        # do not apply total_time_limit
        if self._model_time_limit is not None:
            return True
        # no total time limit, just train, dont ask
        if self._total_time_limit is None:
            return True

        total_time_spend = time.time() - self._start_time
        time_left = self._total_time_limit - total_time_spend
        # no time left, do not train any more models, sorry ...
        if time_left <= 0:
            # print("No time left", time_left)
            return False

        # check the fit level type
        # we dont want to spend too much time on one step
        if not self.enough_time_for_step(step):
            # print("Not enough time for step", step)
            return False

        # there is still time and model_type was not tested yet
        # we should try it
        if expected_time is None and self.model_spend(model_type) == 0:
            return True

        # stacked models converge faster
        # dont need to check ...
        if step == "stack":
            return True
        # check if there is enough time for model to train
        return self.enough_time_for_model(model_type, step, expected_time)

    def learner_time_limit(self, model_type, fit_level, k_folds):
        if self._total_time_limit is None:
            return 7 * 24 * 3600

        if self._model_time_limit is not None:
            return self._model_time_limit / k_folds

        # just train them ...
        if fit_level == "simple_algorithms":
            return None
        if fit_level == "default_algorithms":
            return None

        if self._active_step == fit_level and self._active_algorithms is not None:
            if model_type not in self._active_algorithms:
                return 0.0
            allowance = self.time_should_use(fit_level) / len(self._active_algorithms)
            return max(0.0, min(allowance, self.remaining_step_time(fit_level))) / k_folds

        tune_algorithms = [
            a
            for a in self._algorithms
            if a not in ["Baseline", "Linear", "Decision Tree", "Nearest Neighbors"]
        ]
        tune_algs_cnt = len(tune_algorithms)
        if tune_algs_cnt == 0:
            return None

        time_elapsed = time.time() - self._start_time
        time_left = self._total_time_limit - time_elapsed

        if fit_level == "not_so_random":
            tt = self.time_should_use(fit_level)

            tt /= tune_algs_cnt  # give time equally for each algorithm
            tt /= k_folds  # time is per learner (per fold)
            return tt

        if "hill_climbing" in fit_level:
            tt = self.time_should_use(fit_level)
            tt /= tune_algs_cnt  # give time equally for each algorithm
            tt /= k_folds  # time is per learner (per fold)
            return tt

        if self._is_stacking and fit_level == "stack":
            tt = time_left
            tt /= tune_algs_cnt  # give time equally for each algorithm
            tt /= k_folds  # time is per learner (per fold)
            return tt

    def log_time(self, model_name, model_type, fit_level, train_time):
        self._spend += [
            {
                "model_name": model_name,
                "model_type": model_type,
                "fit_level": fit_level,
                "train_time": train_time,
            }
        ]
        # print(pd.DataFrame(self._spend))
        # print("Already spend", self.already_spend())

    def step_spend(self, step):
        return np.sum([s["train_time"] for s in self._spend if s["fit_level"] == step])

    def model_spend(self, model_type, step=None):
        return np.sum(
            [s["train_time"] for s in self._spend if s["model_type"] == model_type
             and (step is None or s["fit_level"] == step)]
        )
