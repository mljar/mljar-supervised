import os
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


class TunerHillClimbingTest(unittest.TestCase):
    def test_hill_climbing(self):

        models = []
        models += [
            ModelMock(
                "model_121",
                "Random Forest",
                0.1,
                {
                    "learner": {"max_features": 0.4, "model_type": "Random Forest"},
                    "preprocessing": {},
                },
            )
        ]
        models += [
            ModelMock(
                "model_1",
                "Random Forest",
                0.1,
                {
                    "learner": {"max_features": 0.4, "model_type": "Random Forest"},
                    "preprocessing": {},
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
            validation={},
            seed=12,
        )
        ind = 121
        score = 0.1
        for _ in range(5):
            for p in tuner.get_hill_climbing_params(models):
                models += [ModelMock(p["name"], "Random Forest", score, p)]
                score *= 0.1
                self.assertTrue(int(p["name"].split("_")[1]) > ind)
                ind += 1
