import pandas as pd
from supervised.automl import AutoML

df = pd.read_csv("https://raw.githubusercontent.com/pplonski/datasets-for-start/master/adult/data.csv", skipinitialspace=True)

X = df[df.columns[:-1]]
y = df["income"]

automl = AutoML(total_time_limit=10)
automl.fit(X, y)

predictions = automl.predict(X)

print(predictions.head())