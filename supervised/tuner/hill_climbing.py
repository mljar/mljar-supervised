import numpy as np
import copy
from supervised.tuner.registry import ModelsRegistry
from supervised.tuner.registry import BINARY_CLASSIFICATION

class HillClimbing:

    """
    Example params are in JSON format:
    {
        "booster": ["gbtree", "gblinear"],
        "objective": ["binary:logistic"],
        "eval_metric": ["auc", "logloss"],
        "eta": [0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.075, 0.1]
    }
    """

    @staticmethod
    def get(params, seed=None):
        np.random.seed(seed)
        print("---hill---climbing---")
        keys = list(params.keys())
        print(keys)
        keys.remove("model_type")
        print(keys, params)
        key_to_update = np.random.permutation(keys)[0]
        print("key_to_update", key_to_update)

        model_type = params["model_type"]
        model_info = ModelsRegistry.registry[BINARY_CLASSIFICATION][model_type]
        model_params = model_info["params"]
        print("model_params", model_params)
        left, right = None, None
        values = model_params[key_to_update]
        for i, v in enumerate(values):
            print("v", v)
            if v == params[key_to_update]:
                print("***")
                if i+1 < len(values):
                    right = values[i+1]
                if i-1 >= 0:
                    left = values[i-1]

        params_1, params_2 = None, None
        if left is not None:
            params_1 = copy.deepcopy(params)
