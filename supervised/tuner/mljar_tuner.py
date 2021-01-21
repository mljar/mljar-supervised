import os
import copy
import json
import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder

from supervised.tuner.random_parameters import RandomParameters
from supervised.algorithms.registry import AlgorithmsRegistry
from supervised.preprocessing.preprocessing_categorical import PreprocessingCategorical
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
        validation_strategy,
        explain_level,
        data_info,
        golden_features,
        features_selection,
        train_ensemble,
        stack_models,
        adjust_validation,
        boost_on_errors,
        kmeans_features,
        seed,
    ):
        logger.debug("MljarTuner.__init__")
        self._start_random_models = tuner_params.get("start_random_models", 5)
        self._hill_climbing_steps = tuner_params.get("hill_climbing_steps", 3)
        self._top_models_to_improve = tuner_params.get("top_models_to_improve", 3)
        self._algorithms = algorithms
        self._ml_task = ml_task
        self._validation_strategy = validation_strategy
        self._explain_level = explain_level
        self._data_info = data_info
        self._golden_features = golden_features
        self._features_selection = features_selection
        self._train_ensemble = train_ensemble
        self._stack_models = stack_models
        self._adjust_validation = adjust_validation
        self._boost_on_errors = boost_on_errors
        self._kmeans_features = kmeans_features
        self._seed = seed

        self._unique_params_keys = []

    def _apply_categorical_strategies(self):
        if self._data_info is None:
            return []
        if self._data_info.get("columns_info") is None:
            return []

        strategies = []
        for k, v in self._data_info["columns_info"].items():
            # if (
            #    "categorical" in v
            #    and PreprocessingTuner.CATEGORICALS_LOO not in strategies
            # ):
            #    strategies += [PreprocessingTuner.CATEGORICALS_LOO]

            if (
                PreprocessingCategorical.FEW_CATEGORIES in v
                and PreprocessingTuner.CATEGORICALS_MIX not in strategies
            ):
                strategies += [PreprocessingTuner.CATEGORICALS_MIX]

            if len(strategies) == 1:  # disable loo encoding
                # cant add more
                # stop
                break

        return strategies

    def _can_apply_kmeans_features(self):
        if self._data_info is None:
            return False

        # are there any continous
        continous_cols = 0
        for k, v in self._data_info["columns_info"].items():
            if "categorical" not in v:
                continous_cols += 1
        
        # too little columns
        if continous_cols == 0:
            return False

        # too many columns
        if continous_cols > 300:
            return False

        # all good, can apply kmeans
        return True

    def _can_apply_golden_features(self):
        if self._data_info is None:
            return False

        # are there any continous
        continous_cols = 0
        for k, v in self._data_info["columns_info"].items():
            if "categorical" not in v:
                continous_cols += 1
        
        # too little columns
        if continous_cols == 0:
            return False

        # all good, can apply golden features
        return True

    def steps(self):

        all_steps = []
        if self._adjust_validation:
            all_steps += ["adjust_validation"]

        all_steps += ["simple_algorithms", "default_algorithms"]
        
        if self._start_random_models > 1:
            all_steps += ["not_so_random"]

        categorical_strategies = self._apply_categorical_strategies()
        if PreprocessingTuner.CATEGORICALS_MIX in categorical_strategies:
            all_steps += ["mix_encoding"]
        if PreprocessingTuner.CATEGORICALS_LOO in categorical_strategies:
            all_steps += ["loo_encoding"]
        if self._golden_features and self._can_apply_golden_features():
            all_steps += ["golden_features"]
        if self._kmeans_features and self._can_apply_kmeans_features():
            all_steps += ["kmeans_features"]
        if self._features_selection:
            all_steps += ["insert_random_feature"]
            all_steps += ["features_selection"]
        for i in range(self._hill_climbing_steps):
            all_steps += [f"hill_climbing_{i+1}"]
        if self._boost_on_errors:
            all_steps += ["boost_on_errors"]
        if self._train_ensemble:
            all_steps += ["ensemble"]
        if self._stack_models:
            all_steps += ["stack"]
            if self._train_ensemble:
                all_steps += ["ensemble_stacked"]
        return all_steps

    def get_model_name(self, model_type, models_cnt, special=""):
        return f"{models_cnt}_" + special + model_type.replace(" ", "")

    def filter_random_feature_model(self, models):
        return [m for m in models if "RandomFeature" not in m.get_name()]

    def generate_params(
        self, step, models, results_path, stacked_models, total_time_limit
    ):
        try:
            models_cnt = len(models)
            if step == "adjust_validation":
                return self.adjust_validation_params(models_cnt)
            elif step == "simple_algorithms":
                return self.simple_algorithms_params(models_cnt)
            elif step == "default_algorithms":
                return self.default_params(models_cnt)
            elif step == "not_so_random":
                return self.get_not_so_random_params(models_cnt)
            elif step == "mix_encoding":
                return self.get_mix_categorical_strategy(models, total_time_limit)
            elif step == "loo_encoding":
                return self.get_loo_categorical_strategy(models, total_time_limit)
            elif step == "golden_features":
                return self.get_golden_features_params(
                    models, results_path, total_time_limit
                )
            elif step == "kmeans_features":
                return self.get_kmeans_features_params(
                    models, results_path, total_time_limit
                )
            elif step == "insert_random_feature":
                return self.get_params_to_insert_random_feature(
                    models, total_time_limit
                )
            elif step == "features_selection":
                return self.get_features_selection_params(
                    self.filter_random_feature_model(models),
                    results_path,
                    total_time_limit,
                )
            elif "hill_climbing" in step:
                return self.get_hill_climbing_params(
                    self.filter_random_feature_model(models)
                )
            elif step == "boost_on_errors":
                return self.boost_params(models, results_path)
            elif step == "ensemble":
                return [
                    {
                        "model_type": "ensemble",
                        "is_stacked": False,
                        "name": "Ensemble",
                        "status": "initialized",
                        "final_loss": None,
                        "train_time": None,
                    }
                ]
            elif step == "stack":
                return self.get_params_stack_models(stacked_models)
            elif step == "ensemble_stacked":

                # do we have stacked models?
                any_stacked = False
                for m in models:
                    if m._is_stacked:
                        any_stacked = True
                if not any_stacked:
                    return []

                return [
                    {
                        "model_type": "ensemble",
                        "is_stacked": True,
                        "name": "Ensemble_Stacked",
                        "status": "initialized",
                        "final_loss": None,
                        "train_time": None,
                    }
                ]

            # didnt find anything matching the step, return empty array
            return []
        except Exception as e:
            return []

    def get_params_stack_models(self, stacked_models):
        if stacked_models is None or len(stacked_models) == 0:
            return []

        X_train_stacked_path = ""
        added_columns = []

        generated_params = []
        # resue old params
        for m in stacked_models:
            # use only Xgboost, LightGBM and CatBoost as stacked models
            if m.get_type() not in ["Xgboost", "LightGBM", "CatBoost"]:
                continue

            if m.params.get("injected_sample_weight", False):
                # dont use boost_on_errors model for stacking
                # there will be additional boost_on_errors step
                continue

            params = copy.deepcopy(m.params)

            params["validation_strategy"]["X_path"] = params["validation_strategy"][
                "X_path"
            ].replace("X.parquet", "X_stacked.parquet")

            params["name"] = params["name"] + "_Stacked"
            params["is_stacked"] = True
            params["status"] = "initialized"
            params["final_loss"] = None
            params["train_time"] = None

            if "model_architecture_json" in params["learner"]:
                # the new model will be created with wider input size
                del params["learner"]["model_architecture_json"]

            if self._ml_task == REGRESSION:
                # scale added predictions in regression if the target was scaled (in the case of NN)
                # this piece of code might not work, leave it as it is, because NN is not used for training with Stacked Data
                target_preprocessing = params["preprocessing"]["target_preprocessing"]
                scale = None
                if "scale_log_and_normal" in target_preprocessing:
                    scale = "scale_log_and_normal"
                elif "scale_normal" in target_preprocessing:
                    scale = "scale_normal"
                if scale is not None:
                    for col in added_columns:
                        params["preprocessing"]["columns_preprocessing"][col] = [scale]

            generated_params += [params]
        return generated_params

    def adjust_validation_params(self, models_cnt):
        generated_params = []
        for model_type in ["Decision Tree"]:
            models_to_check = 1

            logger.info(f"Generate parameters for {model_type} (#{models_cnt + 1})")
            params = self._get_model_params(model_type, seed=1)
            if params is None:
                continue

            params["name"] = self.get_model_name(model_type, models_cnt + 1)
            params["status"] = "initialized"
            params["final_loss"] = None
            params["train_time"] = None

            unique_params_key = MljarTuner.get_params_key(params)
            if unique_params_key not in self._unique_params_keys:
                generated_params += [params]
                models_cnt += 1
        return generated_params

    def simple_algorithms_params(self, models_cnt):
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
                params["status"] = "initialized"
                params["final_loss"] = None
                params["train_time"] = None

                unique_params_key = MljarTuner.get_params_key(params)
                if unique_params_key not in self._unique_params_keys:
                    generated_params += [params]
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
            "LightGBM",
            "Xgboost",
            "CatBoost",
            "Neural Network",
            "Random Forest",
            "Extra Trees",
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
            params["status"] = "initialized"
            params["final_loss"] = None
            params["train_time"] = None

            unique_params_key = MljarTuner.get_params_key(params)
            if unique_params_key not in self._unique_params_keys:
                generated_params += [params]
                models_cnt += 1
        return generated_params

    def get_not_so_random_params(self, models_cnt):

        model_types = [
            "Xgboost",
            "LightGBM",
            "CatBoost",
            "Random Forest",
            "Extra Trees",
            "Neural Network",
            "Nearest Neighbors",
        ]

        generated_params = {m: [] for m in model_types}

        for model_type in model_types:
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
                params["status"] = "initialized"
                params["final_loss"] = None
                params["train_time"] = None

                unique_params_key = MljarTuner.get_params_key(params)
                if unique_params_key not in self._unique_params_keys:
                    generated_params[model_type] += [params]
                    models_cnt += 1
        
        """
        return_params = []
        for i in range(100):
            total = 0
            for m in ["Xgboost", "LightGBM", "CatBoost"]:
                if generated_params[m]:
                    return_params += [generated_params[m].pop(0)]
                total += len(generated_params[m])
            if total == 0:
                break

        rest_params = []
        for m in [
            "Random Forest",
            "Extra Trees",
            "Neural Network",
            "Nearest Neighbors",
        ]:
            rest_params += generated_params[m]
        if rest_params:
            np.random.shuffle(rest_params)
            return_params += rest_params
        """
        return_params = []
        for i in range(100):
            total = 0
            for m in [
                "LightGBM",
                "Xgboost",
                "CatBoost",
                "Random Forest",
                "Extra Trees",
                "Neural Network",
                "Nearest Neighbors",
            ]:
                if generated_params[m]:
                    return_params += [generated_params[m].pop(0)]
                total += len(generated_params[m])
            if total == 0:
                break

        return return_params

    def get_hill_climbing_params(self, current_models):
        df_models, algorithms = self.df_models_algorithms(current_models)
        generated_params = []
        counts = {model_type: 0 for model_type in algorithms}

        for i in range(df_models.shape[0]):

            model_type = df_models["model_type"].iloc[i]
            counts[model_type] += 1
            if counts[model_type] > self._top_models_to_improve:
                continue

            m = df_models["model"].iloc[i]

            for p in HillClimbing.get(
                m.params.get("learner"), self._ml_task, len(current_models) + self._seed
            ):

                model_indices = [
                    int(m.get_name().split("_")[0]) for m in current_models
                ]
                model_max_index = np.max(model_indices)

                logger.info(
                    "Hill climbing step, for model #{0}".format(model_max_index + 1)
                )
                if p is not None:
                    all_params = copy.deepcopy(m.params)
                    all_params["learner"] = p

                    all_params["name"] = self.get_model_name(
                        all_params["learner"]["model_type"],
                        model_max_index + 1 + len(generated_params),
                    )

                    if "golden_features" in all_params["preprocessing"]:
                        all_params["name"] += "_GoldenFeatures"
                    if "drop_features" in all_params["preprocessing"] and len(
                        all_params["preprocessing"]["drop_features"]
                    ):
                        all_params["name"] += "_SelectedFeatures"
                    all_params["status"] = "initialized"
                    all_params["final_loss"] = None
                    all_params["train_time"] = None
                    unique_params_key = MljarTuner.get_params_key(all_params)
                    
                    if unique_params_key not in self._unique_params_keys:
                        generated_params += [all_params]
                    
        return generated_params

    def get_all_int_categorical_strategy(self, current_models, total_time_limit):
        return self.get_categorical_strategy(
            current_models, PreprocessingTuner.CATEGORICALS_ALL_INT, total_time_limit
        )

    def get_mix_categorical_strategy(self, current_models, total_time_limit):
        return self.get_categorical_strategy(
            current_models, PreprocessingTuner.CATEGORICALS_MIX, total_time_limit
        )

    def get_loo_categorical_strategy(self, current_models, total_time_limit):
        return self.get_categorical_strategy(
            current_models, PreprocessingTuner.CATEGORICALS_LOO, total_time_limit
        )

    def get_categorical_strategy(self, current_models, strategy, total_time_limit):

        df_models, algorithms = self.df_models_algorithms(
            current_models, time_limit=0.1 * total_time_limit
        )
        generated_params = []
        for m_type in algorithms:
            # try to add categorical strategy only for below algorithms
            if m_type not in [
                "Xgboost",
                "LightGBM",
                # "Neural Network",
                # "Random Forest",
                # "Extra Trees",
            ]:
                continue
            models = df_models[df_models.model_type == m_type]["model"]

            for i in range(min(1, len(models))):
                m = models.iloc[i]

                params = copy.deepcopy(m.params)
                cols_preprocessing = params["preprocessing"]["columns_preprocessing"]

                for col, preproc in params["preprocessing"][
                    "columns_preprocessing"
                ].items():
                    new_preproc = []
                    convert_categorical = False

                    for p in preproc:
                        if "categorical" not in p:
                            new_preproc += [p]
                        else:
                            convert_categorical = True

                    col_data_info = self._data_info["columns_info"].get(col)
                    few_categories = False
                    if col_data_info is not None and "few_categories" in col_data_info:
                        few_categories = True

                    if convert_categorical:
                        if strategy == PreprocessingTuner.CATEGORICALS_ALL_INT:
                            new_preproc += [PreprocessingCategorical.CONVERT_INTEGER]
                        elif strategy == PreprocessingTuner.CATEGORICALS_LOO:
                            new_preproc += [PreprocessingCategorical.CONVERT_LOO]
                        elif strategy == PreprocessingTuner.CATEGORICALS_MIX:
                            if few_categories:
                                new_preproc += [
                                    PreprocessingCategorical.CONVERT_ONE_HOT
                                ]
                            else:
                                new_preproc += [
                                    PreprocessingCategorical.CONVERT_INTEGER
                                ]

                    cols_preprocessing[col] = new_preproc

                params["preprocessing"]["columns_preprocessing"] = cols_preprocessing
                # if there is already a name of categorical strategy in the name
                # please remove it to avoid confusion (I hope!)
                for st in [
                    PreprocessingTuner.CATEGORICALS_LOO,
                    PreprocessingTuner.CATEGORICALS_ALL_INT,
                    PreprocessingTuner.CATEGORICALS_MIX,
                ]:
                    params["name"] = params["name"].replace("_" + st, "")
                params["name"] += f"_{strategy}"
                params["status"] = "initialized"
                params["final_loss"] = None
                params["train_time"] = None
                if "model_architecture_json" in params["learner"]:
                    del params["learner"]["model_architecture_json"]
                unique_params_key = MljarTuner.get_params_key(params)
                if unique_params_key not in self._unique_params_keys:
                    generated_params += [params]
        return generated_params

    def df_models_algorithms(self, current_models, time_limit=None, exclude_golden=False):
        scores = [m.get_final_loss() for m in current_models]
        model_types = [m.get_type() for m in current_models]
        names = [m.get_name() for m in current_models]
        train_times = [m.get_train_time() for m in current_models]

        df_models = pd.DataFrame(
            {
                "model": current_models,
                "score": scores,
                "model_type": model_types,
                "name": names,
                "train_time": train_times,
            }
        )
        if time_limit is not None:
            df_models = df_models[df_models.train_time < time_limit]
            df_models.reset_index(drop=True, inplace=True)

        if exclude_golden:
            ii = df_models["name"].apply(lambda x: "GoldenFeatures" in x)
            df_models = df_models[~ii]
            df_models.reset_index(drop=True, inplace=True)

        df_models.sort_values(by="score", ascending=True, inplace=True)
        model_types = list(df_models.model_type)
        u, idx = np.unique(model_types, return_index=True)
        algorithms = u[np.argsort(idx)]

        return df_models, algorithms

    def get_golden_features_params(
        self, current_models, results_path, total_time_limit
    ):

        df_models, algorithms = self.df_models_algorithms(
            current_models, time_limit=0.1 * total_time_limit
        )

        generated_params = []
        for i in range(min(3, df_models.shape[0])):
            m = df_models["model"].iloc[i]

            params = copy.deepcopy(m.params)
            params["preprocessing"]["golden_features"] = {
                "results_path": results_path,
                "ml_task": self._ml_task,
            }
            params["name"] += "_GoldenFeatures"
            params["status"] = "initialized"
            params["final_loss"] = None
            params["train_time"] = None

            if "model_architecture_json" in params["learner"]:
                del params["learner"]["model_architecture_json"]
            unique_params_key = MljarTuner.get_params_key(params)
            if unique_params_key not in self._unique_params_keys:
                generated_params += [params]
        return generated_params

    def get_kmeans_features_params(
        self, current_models, results_path, total_time_limit
    ):

        df_models, algorithms = self.df_models_algorithms(
            current_models, time_limit=0.1 * total_time_limit,
            exclude_golden=True
        )

        generated_params = []
        for i in range(min(3, df_models.shape[0])):
            m = df_models["model"].iloc[i]

            params = copy.deepcopy(m.params)
            params["preprocessing"]["kmeans_features"] = {"results_path": results_path}
            params["name"] += "_KMeansFeatures"
            params["status"] = "initialized"
            params["final_loss"] = None
            params["train_time"] = None

            if "model_architecture_json" in params["learner"]:
                del params["learner"]["model_architecture_json"]
            unique_params_key = MljarTuner.get_params_key(params)
            if unique_params_key not in self._unique_params_keys:
                generated_params += [params]
        return generated_params

    def time_features_selection(self, current_models, total_time_limit):

        df_models, algorithms = self.df_models_algorithms(
            current_models, time_limit=0.1 * total_time_limit
        )

        time_needed = 0
        for m_type in algorithms:

            if m_type not in [
                "Xgboost",
                "LightGBM",
                "CatBoost",
                "Neural Network",
                "Random Forest",
                "Extra Trees",
            ]:
                continue
            models = df_models[df_models.model_type == m_type]["model"]

            for i in range(min(1, len(models))):
                m = models.iloc[i]
                if time_needed == 0:
                    # best model will be used two times
                    # one for insert random feature
                    # one for selected features
                    time_needed += 2.0 * m.get_train_time()
                else:
                    time_needed += m.get_train_time()
                

        return time_needed

    def get_params_to_insert_random_feature(self, current_models, total_time_limit):

        time_needed = self.time_features_selection(current_models, total_time_limit)

        if time_needed > 0.1 * total_time_limit:
            print("Not enough time to perform features selection. Skip")
            print(
                "Time needed for features selection ~", np.round(time_needed), "seconds"
            )
            print(
                f"Please increase total_time_limit to at least ({int(np.round(10.0*time_needed))+60} seconds) to have features selection"
            )
            return None

        df_models, algorithms = self.df_models_algorithms(
            current_models, time_limit=0.1 * total_time_limit
        )
        if df_models.shape[0] == 0:
            return None

        m = df_models.iloc[0]["model"]

        params = copy.deepcopy(m.params)
        params["preprocessing"]["add_random_feature"] = True
        params["name"] += "_RandomFeature"
        params["status"] = "initialized"
        params["final_loss"] = None
        params["train_time"] = None
        params["explain_level"] = 1
        if "model_architecture_json" in params["learner"]:
            del params["learner"]["model_architecture_json"]

        unique_params_key = MljarTuner.get_params_key(params)
        if unique_params_key not in self._unique_params_keys:
            return [params]
        return None

    def get_features_selection_params(
        self, current_models, results_path, total_time_limit
    ):

        fname = os.path.join(results_path, "drop_features.json")
        if not os.path.exists(fname):
            return None

        drop_features = json.load(open(fname, "r"))
        print("Drop features", drop_features)

        # in case of droping only one feature (random_feature)
        # skip this step
        if len(drop_features) <= 1:
            return None

        df_models, algorithms = self.df_models_algorithms(
            current_models, time_limit=0.1 * total_time_limit
        )

        generated_params = []
        for m_type in algorithms:
            # try to do features selection only for below algorithms
            if m_type not in [
                "Xgboost",
                "LightGBM",
                "CatBoost",
                "Neural Network",
                "Random Forest",
                "Extra Trees",
            ]:
                continue
            models = df_models[df_models.model_type == m_type]["model"]

            for i in range(min(1, len(models))):
                m = models.iloc[i]

                params = copy.deepcopy(m.params)
                params["preprocessing"]["drop_features"] = drop_features
                params["name"] += "_SelectedFeatures"
                params["status"] = "initialized"
                params["final_loss"] = None
                params["train_time"] = None
                if "model_architecture_json" in params["learner"]:
                    del params["learner"]["model_architecture_json"]
                unique_params_key = MljarTuner.get_params_key(params)
                if unique_params_key not in self._unique_params_keys:
                    generated_params += [params]
        return generated_params

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
            "validation_strategy": self._validation_strategy,
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
        for main_key in ["preprocessing", "learner", "validation_strategy"]:
            key += "_" + main_key
            for k in sorted(params[main_key]):
                if k in ["seed", "explain_level"]:
                    continue
                key += "_{}_{}".format(k, params[main_key][k])
        return key

    def add_key(self, model):
        if model.get_type() != "Ensemble":
            key = MljarTuner.get_params_key(model.params)
            self._unique_params_keys += [key]


    def boost_params(self, current_models, results_path):

        df_models, algorithms = self.df_models_algorithms(current_models)
        best_model = None
        for i in range(df_models.shape[0]):
            if df_models["model_type"].iloc[i] in [
                "Ensemble",
                "Neural Network",
                "Nearest Neighbors",
            ]:
                continue
            best_model = df_models["model"].iloc[i]
            break
        if best_model is None:
            return []

        # load predictions
        oof = best_model.get_out_of_folds()

        predictions = oof[[c for c in oof.columns if c.startswith("prediction")]]
        y = oof["target"]

        if self._ml_task == MULTICLASS_CLASSIFICATION:
            oh = OneHotEncoder(sparse=False)
            y_encoded = oh.fit_transform(np.array(y).reshape(-1, 1))
            residua = np.sum(
                np.abs(np.array(y_encoded) - np.array(predictions)), axis=1
            )
        else:
            residua = np.abs(np.array(y) - np.array(predictions).ravel())

        df_preds = pd.DataFrame(
            {"res": residua, "lp": range(residua.shape[0]), "target": np.array(y)}
        )

        df_preds = df_preds.sort_values(by="res", ascending=True)
        df_preds["order"] = range(residua.shape[0])
        df_preds["order"] = (df_preds["order"]) / residua.shape[0] / 5.0 + 0.9
        df_preds = df_preds.sort_values(by="lp", ascending=True)

        sample_weight_path = os.path.join(
            results_path, best_model.get_name() + "_sample_weight.parquet"
        )
        pd.DataFrame({"sample_weight": df_preds["order"]}).to_parquet(
            sample_weight_path, index=False
        )

        generated_params = []

        params = copy.deepcopy(best_model.params)
        
        params["validation_strategy"]["sample_weight_path"] = sample_weight_path
        params["injected_sample_weight"] = True
        params["name"] += "_BoostOnErrors"
        params["status"] = "initialized"
        params["final_loss"] = None
        params["train_time"] = None
        if "model_architecture_json" in params["learner"]:
            del params["learner"]["model_architecture_json"]
        unique_params_key = MljarTuner.get_params_key(params)

        if unique_params_key not in self._unique_params_keys:
            generated_params += [params]

        return generated_params
