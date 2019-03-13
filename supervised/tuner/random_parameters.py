import numpy as np


class RandomParameters:

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
        generated_params = {}
        for k in params:
            generated_params[k] = np.random.permutation(params[k])[0]
        return generated_params
