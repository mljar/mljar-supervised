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

        ratios = {
            "default_algorithms": 0.3,
            "not_so_random": 0.3,
            "mix_encoding": 0.05,
            "golden_features": 0.05,
            "kmeans_features": 0.05,
            "insert_random_feature": 0.05,
            "features_selection": 0.05,
            "hill_climbing_1": 0.2,  # enough to have only first step from hill climbing
            "boost_on_errors": 0.05,
            "stack": 0.15,
        }

        if (
            fit_level
            in [
                "default_algorithms",
                "not_so_random",
                "boost_on_errors",
                "mix_encoding",
                "golden_features",
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
                #"default_algorithms",
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

    def enough_time_for_step(self, fit_level):

        total_time_spend = time.time() - self._start_time
        compound = self.compound_time_should_use(fit_level)
        #print(fit_level, total_time_spend, compound, self._total_time_limit)
        if total_time_spend > compound:
            # dont train more
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

        # check the fit level type
        # we dont want to spend too much time on one step
        if not self.enough_time_for_step(step):
            # print("Not enough time for step", step)
            return False

        # there is still time and model_type was not tested yet
        # we should try it
        if time_left > 0 and self.model_spend(model_type) == 0:
            return True
        
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
