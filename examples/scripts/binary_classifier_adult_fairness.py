import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from supervised.automl import AutoML
from sklearn.datasets import fetch_openml


# df = pd.read_csv(
#    "https://raw.githubusercontent.com/pplonski/datasets-for-start/master/adult/data.csv",
#    skipinitialspace=True,
# )
# X = df[df.columns[:-1]]
# y = (df["income"] == ">50K") * 1

data = fetch_openml(data_id=1590, as_frame=True)
y = (data.target == ">50K") * 1

X = data.data
y = (data.target == ">50K") * 1

X["is_young"] = (X["age"] < 50) * 1 # < 30
X["is_young"] = X["is_young"].astype(str)


sensitive_features = X[["sex", "age"]] #, "race"]] #, "is_young"]]  # , "race", "age"]]
print("Input data")
print(X)
print("Sensitive features")

print(sensitive_features)


X_train, X_test, y_train, y_test, S_train, S_test = train_test_split(
    X, y, sensitive_features, stratify=y, test_size=0.5, random_state=42
)



automl = AutoML(algorithms=["Xgboost"], # ["Linear", "Xgboost", "LightGBM", "Random Forest", "Decision Tree", "CatBoost"],
                train_ensemble=True,
                fairness_metric="demographic_parity_ratio",  # 
                fairness_threshold=0.8,
                #privileged_groups = [{"sex": "Male"}],
                #underprivileged_groups = [{"sex": "Female"}],
                #hill_climbing_steps=1,
                #top_models_to_improve=1,
                explain_level=1
            )

automl.fit(X_train, y_train, sensitive_features=S_train)




# Example
#
# Fairness training with custom validation strategy
#
# automl = AutoML(algorithms=["Xgboost"], 
#                 train_ensemble=False,
#                 fairness_metric="demographic_parity_ratio",  # 
#                 fairness_threshold=0.8,
#                 hill_climbing_steps=1,
#                 top_models_to_improve=1,
#                 validation_strategy={"validation_type": "custom"},
#                 explain_level=1
#             )
# train_indices = np.array(list(range(0, X_train.shape[0]//2)))
# test_indices = np.array(list(range(X_train.shape[0]//2, X_train.shape[0])))
# cv = [(train_indices, test_indices)]
# automl.fit(X_train, y_train, sensitive_features=S_train, cv=cv)


# Example
# 
# Fairness training with cross validation
# 
# automl = AutoML(algorithms=["Xgboost"],
#                 train_ensemble=False,
#                 fairness_metric="demographic_parity_ratio",  # 
#                 fairness_threshold=0.8,
#                 hill_climbing_steps=1,
#                 top_models_to_improve=1,
#                 validation_strategy={
#                     "validation_type": "kfold",
#                     "k_folds": 5,
#                     "shuffle": True,
#                     "stratify": True,
#                     "random_seed": 123
#                 },
#                 explain_level=1
#             )
# automl.fit(X_train, y_train, sensitive_features=S_train)
