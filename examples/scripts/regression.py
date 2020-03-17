import pandas as pd
from supervised.automl import AutoML


df = pd.read_csv("./tests/data/housing_regression_missing_values_missing_target.csv")
print(df.columns)
x_cols = [c for c in df.columns if c != "MEDV"]
X = df[x_cols]
y = df["MEDV"]


print(X)
print(y)

automl = AutoML(total_time_limit=10)
automl.fit(X, y)

predictions = automl.predict(X)

print(predictions.head())
