import pandas as pd
from supervised.automl import AutoML


df = pd.read_csv("./tests/data/housing_regression_missing_values_missing_target.csv")
x_cols = [c for c in df.columns if c != "MEDV"]
X = df[x_cols]
y = df["MEDV"]

automl = AutoML(total_time_limit=1000, start_random_models=5,
        hill_climbing_steps=0,
        top_models_to_improve=0)
automl.fit(X, y)


df["predictions"] = automl.predict(X)
print("Predictions")
print(df[["MEDV", "predictions"]].head())
