import numpy as np
import pandas as pd
from supervised.automl import AutoML

# data source http://archive.ics.uci.edu/ml/datasets/Communities%20and%20Crime%20Unnormalized

df = pd.read_csv("tests/data/LawSchool/bar_pass_prediction.csv")
df["race1"][df["race1"] != "white"] = "non-white"

print(df)
print(df.shape)
X = df[["gender", "lsat", "race1", "pass_bar"]] #df.columns[3:36]]
print(X)
y = df["gpa"]
print(y)


sensitive_features = df["race1"]
print(sensitive_features)
print(df["race1"].unique())



automl = AutoML(algorithms=["Xgboost"], train_ensemble=False, fairness_threshold=0.95)
automl.fit(X, y, sensitive_features=sensitive_features)

# df["predictions"] = automl.predict(X)
# print("Predictions")
# print(df[["MEDV", "predictions"]].head())
