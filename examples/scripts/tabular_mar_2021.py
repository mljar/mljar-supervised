import pandas as pd
from supervised import AutoML

train = pd.read_csv("~/Downloads/tabular-playground-series-mar-2021/train.csv")
test = pd.read_csv("~/Downloads/tabular-playground-series-mar-2021/test.csv")

X_train = train.drop(["id", "target"], axis=1)
y_train = train.target
X_test = test.drop(["id"], axis=1)

init_params = {
    "original_LightGBM": {
        "num_leaves": 1063,
        "lambda_l1": 9.433292446693898,
        "lambda_l2": 2.4718165404756194,
        "feature_fraction": 0.3896871148317942,
        "bagging_fraction": 0.9993283707173064,
        "bagging_freq": 1,
        "min_data_in_leaf": 17,
        "cat_l2": 39.65807273563681,
        "cat_smooth": 38.79107412232164,
        "metric": "auc",
        "num_boost_round": 1000,
        "early_stopping_rounds": 50,
        "learning_rate": 0.025,
        "cat_feature": [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18
        ],
        "feature_pre_filter": False,
        "seed": 123
    },
    "original_golden_features_LightGBM": {
        "learning_rate": 0.025,
        "num_leaves": 268,
        "lambda_l1": 1.0427167504924784,
        "lambda_l2": 0.0003457339632879811,
        "feature_fraction": 0.3855255799605316,
        "bagging_fraction": 0.9164565505971458,
        "bagging_freq": 2,
        "min_data_in_leaf": 88,
        "cat_l2": 2.377819780894969,
        "cat_smooth": 28.72709046516454,
        "metric": "auc",
        "num_boost_round": 1000,
        "early_stopping_rounds": 50,
        "cat_feature": [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18
        ],
        "feature_pre_filter": False,
        "seed": 123
    }
}

automl = AutoML(mode="Optuna", 
                eval_metric="auc",
                optuna_time_budget=1800,  # tune each algorithm for 30 minutes
                total_time_limit=8*3600)  # total time limit
automl.fit(X_train, y_train)

preds = automl.predict_proba(X_test)
submission = pd.DataFrame({"id": test.id, "target": preds[:, 1]})
submission.to_csv("1_submission.csv", index=False)
