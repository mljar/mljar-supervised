import numpy as np
import pandas as pd
from supervised.automl import AutoML

df = pd.read_csv("./tests/data/boston_housing.csv")
x_cols = [c for c in df.columns if c != "MEDV"]
X = df[x_cols]
y = df["MEDV"]

sensitive_features = df["B"]

automl = AutoML()
automl.fit(X, y, sensitive_features=sensitive_features)

df["predictions"] = automl.predict(X)
print("Predictions")
print(df[["MEDV", "predictions"]].head())
