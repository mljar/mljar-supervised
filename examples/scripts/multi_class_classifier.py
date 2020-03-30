import pandas as pd
from supervised.automl import AutoML
from sklearn.datasets import load_iris

df = pd.read_csv("tests/data/iris_missing_values_missing_target.csv")
X = df[["feature_1","feature_2","feature_3","feature_4"]]
y = df["class"]

print(X)
print(y)

automl = AutoML(total_time_limit=10)
automl.fit(X, y)

predictions = automl.predict(X)

print(predictions.head())
print(predictions.tail())