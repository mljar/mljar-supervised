import pandas as pd
from supervised import AutoML

train = pd.read_csv("~/Downloads/tabular-playground-series-mar-2021/train.csv")
test = pd.read_csv("~/Downloads/tabular-playground-series-mar-2021/test.csv")

X_train = train.drop(["id", "target"], axis=1)
y_train = train.target
X_test = test.drop(["id"], axis=1)

automl = (
    AutoML(  # results_path="/media/piotr/2t/tabular-playground-mar-2021-catboost-3",
        mode="Optuna",
        total_time_limit=48*3600,
        algorithms=[
             "LightGBM", 
             "Xgboost", 
             "CatBoost"
            ], #, "CatBoost" "LightGBM", "CatBoost"], 
        #start_random_models=1,
        #hill_climbing_steps=0,
        eval_metric="auc",
        validation_strategy={
            "validation_type": "kfold",
            "k_folds": 10,
            "shuffle": True,
            "stratify": True,
        },
        random_state=42,
        #mix_encoding=True,
        #kmeans_features=True,
        #golden_features=True,
        #features_selection=False,
        #stack_models=False,
        optuna_time_budget=1800,
        #n_jobs = 18
    )
)
automl.fit(X_train, y_train)

preds = automl.predict_proba(X_test)
submission = pd.DataFrame({"id": test.id, "target": preds[:, 1]})
submission.to_csv("1_submission.csv", index=False)
