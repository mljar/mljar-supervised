import pandas as pd
from supervised.automl import AutoML
from sklearn.datasets import load_iris

iris = load_iris()

X = pd.DataFrame(iris.data)
y = iris.target

print(X)

automl = AutoML(total_time_limit=10)
automl.fit(X, y)

predictions = automl.predict(X)

print(predictions.head())