import numpy as np
import pandas as pd
import copy

from supervised.tuner.random_parameters import RandomParameters
from supervised.algorithms.registry import AlgorithmsRegistry
from supervised.tuner.preprocessing_tuner import PreprocessingTuner
from supervised.tuner.hill_climbing import HillClimbing
from supervised.algorithms.registry import (
    BINARY_CLASSIFICATION,
    MULTICLASS_CLASSIFICATION,
    REGRESSION,
)

import logging
from supervised.utils.config import LOG_LEVEL

logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)


class MljarTuner:
    def __init__(
        self,
        tuner_params,
        algorithms,
        ml_task,
        validation,
        explain_level,
        data_info,
        seed,
    ):
        logger.debug("MljarTuner.__init__")
        self._start_random_models = tuner_params.get("start_random_models", 5)
        self._hill_climbing_steps = tuner_params.get("hill_climbing_steps", 3)
        self._top_models_to_improve = tuner_params.get("top_models_to_improve", 3)
        self._algorithms = algorithms
        self._ml_task = ml_task
        self._validation = validation
        self._explain_level = explain_level
        self._data_info = data_info
        self._seed = seed

        self._unique_params_keys = []

    def get_model_name(self, model_type, models_cnt, special=""):
        return f"{models_cnt}_" + special + model_type.replace(" ", "")

    def simple_algorithms_params(self):
        models_cnt = 0
        generated_params = []
        for model_type in ["Baseline", "Decision Tree", "Linear"]:
            if model_type not in self._algorithms:
                continue
            models_to_check = 1
            if model_type == "Decision Tree":
                models_to_check = min(3, self._start_random_models)
            for i in range(models_to_check):
                logger.info(f"Generate parameters for {model_type} (#{models_cnt + 1})")
                params = self._get_model_params(model_type, seed=i + 1)
                if params is None:
                    continue

                params["name"] = self.get_model_name(model_type, models_cnt + 1)

                unique_params_key = MljarTuner.get_params_key(params)
                if unique_params_key not in self._unique_params_keys:
                    generated_params += [params]
                    self._unique_params_keys += [unique_params_key]
                    models_cnt += 1
        return generated_params

    def skip_if_rows_cols_limit(self, model_type):

        max_rows_limit = AlgorithmsRegistry.get_max_rows_limit(
            self._ml_task, model_type
        )
        max_cols_limit = AlgorithmsRegistry.get_max_cols_limit(
            self._ml_task, model_type
        )

        if max_rows_limit is not None:
            if self._data_info["rows"] > max_rows_limit:
                return True
        if max_cols_limit is not None:
            if self._data_info["cols"] > max_cols_limit:
                return True

        return False

    def default_params(self, models_cnt):

        generated_params = []
        for model_type in [
            "Random Forest",
            "Extra Trees",
            "Xgboost",
            "LightGBM",
            "CatBoost",
            "Neural Network",
            "Nearest Neighbors",
        ]:
            if model_type not in self._algorithms:
                continue

            if self.skip_if_rows_cols_limit(model_type):
                continue

            logger.info(f"Get default parameters for {model_type} (#{models_cnt + 1})")
            params = self._get_model_params(
                model_type, seed=models_cnt + 1, params_type="default"
            )
            if params is None:
                continue
            params["name"] = self.get_model_name(
                model_type, models_cnt + 1, special="Default_"
            )

            unique_params_key = MljarTuner.get_params_key(params)
            if unique_params_key not in self._unique_params_keys:
                generated_params += [params]
                self._unique_params_keys += [unique_params_key]
                models_cnt += 1
        return generated_params

    def get_not_so_random_params(self, models_cnt):

        generated_params = []

        for model_type in [
            "Xgboost",
            "LightGBM",
            "CatBoost",
            "Random Forest",
            "Extra Trees",
            "Neural Network",
            "Nearest Neighbors",
        ]:
            if model_type not in self._algorithms:
                continue

            if self.skip_if_rows_cols_limit(model_type):
                continue
            # minus 1 because already have 1 default
            for i in range(self._start_random_models - 1):

                logger.info(
                    f"Generate not-so-random parameters for {model_type} (#{models_cnt+1})"
                )
                params = self._get_model_params(model_type, seed=i + 1)
                if params is None:
                    continue

                params["name"] = self.get_model_name(model_type, models_cnt + 1)

                unique_params_key = MljarTuner.get_params_key(params)
                if unique_params_key not in self._unique_params_keys:
                    generated_params += [params]
                    self._unique_params_keys += [unique_params_key]
                    models_cnt += 1

        # shuffle params - switch off
        # np.random.shuffle(generated_params)
        return generated_params

    def get_hill_climbing_params(self, current_models):

        # second, hill climbing
        for _ in range(self._hill_climbing_steps):
            # get models orderer by loss
            # TODO: refactor this callbacks.callbacks[0]
            scores = [m.get_final_loss() for m in current_models]
            model_types = [m.get_type() for m in current_models]
            df_models = pd.DataFrame(
                {"model": current_models, "score": scores, "model_type": model_types}
            )
            # do group by for debug reason
            df_models = df_models.groupby("model_type").apply(
                lambda x: x.sort_values("score")
            )
            unique_model_types = np.unique(df_models.model_type)

            for m_type in unique_model_types:
                if m_type in [
                    "Baseline",
                    "Decision Tree",
                    "Linear",
                    "Nearest Neighbors",
                ]:
                    # dont tune Baseline and Decision Tree
                    continue
                models = df_models[df_models.model_type == m_type]["model"]

                for i in range(min(self._top_models_to_improve, len(models))):
                    m = models[i]

                    for p in HillClimbing.get(
                        m.params.get("learner"),
                        self._ml_task,
                        len(current_models) + self._seed,
                    ):

                        model_indices = [
                            int(m.get_name().split("_")[0]) for m in current_models
                        ]
                        model_max_index = np.max(model_indices)

                        logger.info(
                            "Hill climbing step, for model #{0}".format(
                                model_max_index + 1
                            )
                        )
                        if p is not None:
                            all_params = copy.deepcopy(m.params)
                            all_params["learner"] = p

                            all_params["name"] = self.get_model_name(
                                all_params["learner"]["model_type"], model_max_index + 1
                            )

                            unique_params_key = MljarTuner.get_params_key(all_params)
                            if unique_params_key not in self._unique_params_keys:
                                self._unique_params_keys += [unique_params_key]
                                yield all_params

    def _get_model_params(self, model_type, seed, params_type="random"):
        model_info = AlgorithmsRegistry.registry[self._ml_task][model_type]

        model_params = None
        if params_type == "default":

            model_params = model_info["default_params"]
            model_params["seed"] = seed

        else:
            model_params = RandomParameters.get(model_info["params"], seed + self._seed)
        if model_params is None:
            return None

        required_preprocessing = model_info["required_preprocessing"]
        model_additional = model_info["additional"]
        preprocessing_params = PreprocessingTuner.get(
            required_preprocessing, self._data_info, self._ml_task
        )

        model_params = {
            "additional": model_additional,
            "preprocessing": preprocessing_params,
            "validation": self._validation,
            "learner": {
                "model_type": model_info["class"].algorithm_short_name,
                "ml_task": self._ml_task,
                **model_params,
            },
        }

        if self._data_info.get("num_class") is not None:
            model_params["learner"]["num_class"] = self._data_info.get("num_class")

        model_params["ml_task"] = self._ml_task
        model_params["explain_level"] = self._explain_level

        return model_params

    @staticmethod
    def get_params_key(params):
        key = "key_"
        for main_key in ["preprocessing", "learner"]:
            key += main_key
            for k, v in params[main_key].items():
                if k == "seed":
                    continue
                key += "_{}_{}".format(k, v)
        return key
