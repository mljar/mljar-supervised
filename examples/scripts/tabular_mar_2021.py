import pandas as pd
from supervised import AutoML

train = pd.read_csv("~/Downloads/tabular-playground-series-mar-2021/train.csv")
test = pd.read_csv("~/Downloads/tabular-playground-series-mar-2021/test.csv")

X_train = train.drop(["id", "target"], axis=1)
y_train = train.target
X_test = test.drop(["id"], axis=1)

automl = AutoML(
    mode="Optuna",
    eval_metric="auc",
    algorithms=["CatBoost"],
    optuna_time_budget=1800,  # tune each algorithm for 30 minutes
    total_time_limit=48
    * 3600,  # total time limit, set large enough to have time to compute all steps
    features_selection=False,
)
automl.fit(X_train, y_train)

preds = automl.predict_proba(X_test)
submission = pd.DataFrame({"id": test.id, "target": preds[:, 1]})
submission.to_csv("1_submission.csv", index=False)
