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
    def get(params, seed=1):
        np.random.seed(seed)
        generated_params = {"seed": seed}
        for k in params:
            # we need to convert numpy types to native python types
            # it is needed in JSON serialization
            generated_params[k] = np.random.permutation(params[k])[0].item()
        return generated_params
