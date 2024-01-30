import unittest

from supervised.tuner.mljar_tuner import MljarTuner


class ModelMock:
    def __init__(self, name, model_type, final_loss, params):
        self.name = name
        self.model_type = model_type
        self.final_loss = final_loss
        self.params = params

    def get_name(self):
        return self.name

    def get_type(self):
        return self.model_type

    def get_final_loss(self):
        return self.final_loss

    def get_train_time(self):
        return 0.1


class TunerHillClimbingTest(unittest.TestCase):
    def test_hill_climbing(self):
        models = []
        models += [
            ModelMock(
                "121_RandomForest",
                "Random Forest",
                0.1,
                {
                    "learner": {"max_features": 0.4, "model_type": "Random Forest"},
                    "preprocessing": {},
                    "validation_strategy": {},
                },
            )
        ]
        models += [
            ModelMock(
                "1_RandomForest",
                "Random Forest",
                0.1,
                {
                    "learner": {"max_features": 0.4, "model_type": "Random Forest"},
                    "preprocessing": {},
                    "validation_strategy": {},
                },
            )
        ]
        tuner = MljarTuner(
            {
                "start_random_models": 0,
                "hill_climbing_steps": 1,
                "top_models_to_improve": 2,
            },
            algorithms=["Random Foresrt"],
            ml_task="binary_classification",
            eval_metric="logloss",
            validation_strategy={},
            explain_level=2,
            data_info={"columns_info": [], "target_info": []},
            golden_features=False,
            features_selection=False,
            train_ensemble=False,
            stack_models=False,
            adjust_validation=False,
            boost_on_errors=False,
            kmeans_features=False,
            mix_encoding=False,
            optuna_time_budget=None,
            optuna_init_params={},
            optuna_verbose=True,
            n_jobs=1,
            seed=12,
        )
        ind = 121
        score = 0.1
        for _ in range(5):
            for p in tuner.get_hill_climbing_params(models):
                models += [ModelMock(p["name"], "Random Forest", score, p)]
                score *= 0.1
                self.assertTrue(int(p["name"].split("_")[0]) > ind)
                ind += 1
