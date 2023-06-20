import numpy as np
import pandas as pd
from supervised.automl import AutoML

df = pd.read_csv("./tests/data/boston_housing.csv")
x_cols = [c for c in df.columns if c != "MEDV"]

df["large_B"] = (df["B"] > 380) * 1
df["large_B"] = df["large_B"].astype(str)


print(df["large_B"].dtype.name)
sensitive_features = df["large_B"]

X = df[x_cols]
y = df["MEDV"]

automl = AutoML(
    algorithms=["Xgboost", "LightGBM"],
    train_ensemble=True,
    fairness_threshold=0.9,
)
automl.fit(X, y, sensitive_features=sensitive_features)

df["predictions"] = automl.predict(X)
print("Predictions")
print(df[["MEDV", "predictions"]].head())
