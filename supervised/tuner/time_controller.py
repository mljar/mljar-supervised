import time
import json
import logging
import numpy as np
import pandas as pd
from supervised.utils.config import LOG_LEVEL

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

    def to_json(self):
        return {
            "total_time_limit": self._total_time_limit,
            "model_time_limit": self._model_time_limit,
            "steps": self._steps,
            "algorithms": self._algorithms,
            "spend": self._spend,
            "is_hill_climbing": self._is_hill_climbing,
            "is_stacking": self._is_stacking,
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
            tc._start_time -= tc.already_spend()  # update time with already spend
            return tc
        except Exception as e:
            logger.error(f"Cant load TimeController from json, {str(e)}")
            pass
        return None

    def already_spend(self):
        return np.sum([s["train_time"] for s in self._spend])

    def time_should_use(self, fit_level):

        if fit_level == "not_so_random":

            time_should_use = (
                self._total_time_limit
                - self.step_spend("simple_algorithms")
                - self.step_spend("default_algorithms")
            )
            if self._is_stacking:
                time_should_use *= 0.66  # leave time for stacking
            if self._is_hill_climbing:
                time_should_use *= 0.5  # leave time for hill-climbing
            return time_should_use

        if "hill_climbing" in fit_level or fit_level in [
            "golden_features",
            "insert_random_feature",
            "features_selection",
        ]:

            time_should_use = (
                self._total_time_limit
                - self.step_spend("simple_algorithms")
                - self.step_spend("default_algorithms")
                - self.step_spend("not_so_random")
            )
            if self._is_stacking:
                time_should_use *= 0.5  # leave time for stacking
            return time_should_use

        return self._total_time_limit

    def enough_time_for_step(self, fit_level):

        total_time_spend = time.time() - self._start_time
        if fit_level == "not_so_random":

            time_should_use = self.time_should_use(fit_level)

            # print("not_so_random should use", time_should_use)
            # print(total_time_spend)
            # print(
            #    self.step_spend("simple_algorithms")
            #    + self.step_spend("default_algorithms")
            # )

            if total_time_spend > time_should_use + self.step_spend(
                "simple_algorithms"
            ) + self.step_spend("default_algorithms"):
                return False

        if "hill_climbing" in fit_level or fit_level in [
            "golden_features",
            "insert_random_feature",
            "features_selection",
        ]:

            time_should_use = self.time_should_use(fit_level)

            if total_time_spend > time_should_use + self.step_spend(
                "simple_algorithms"
            ) + self.step_spend("default_algorithms") + self.step_spend(
                "not_so_random"
            ):
                return False

        return True

    def enough_time_for_model(self, model_type):

        time_left = self._total_time_limit - self.already_spend()
        spend = [s["train_time"] for s in self._spend if s["model_type"] == model_type]
        model_mean_spend = np.mean(spend)
        return model_mean_spend <= time_left

    def enough_time(self, model_type, step):
        """
        Check if there is enough time to train the next model.
        
        Parameters
        ----------
        model_type : str
            String with type of the model.
        
        step: str
            String with name of the step in the process of AutoML training.


        Returns
        -------
        bool
            `True` if there is time for training next model, `False` otherwise.
        """

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
        if time_left < 0:
            # print("No time left", time_left)
            return False

        # there is still time and model_type was not tested yet
        # we should try it
        if time_left > 0 and self.model_spend(model_type) == 0:
            return True

        # check the fit level type
        # we dont want to spend too much time on one step
        if not self.enough_time_for_step(step):
            # print("Not enough time for step", step)
            return False

        # check if there is enough time for model to train
        return self.enough_time_for_model(model_type)

    def learner_time_limit(self, model_type, fit_level, k_folds):

        if self._model_time_limit is not None:
            return self._model_time_limit / k_folds

        # just train them ...
        if fit_level == "simple_algorithms":
            return None
        if fit_level == "default_algorithms":
            return None

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

    def model_spend(self, model_type):
        return np.sum(
            [s["train_time"] for s in self._spend if s["model_type"] == model_type]
        )
