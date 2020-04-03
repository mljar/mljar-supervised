import pandas as pd
from supervised.automl import AutoML
import os

df = pd.read_csv("https://raw.githubusercontent.com/pplonski/datasets-for-start/master/adult/data.csv", skipinitialspace=True)

X = df[df.columns[:-1]]
y = df["income"]


automl = AutoML(
        #results_path = "AutoML_8",
        total_time_limit=50,
        start_random_models=5,
        hill_climbing_steps=3,
        top_models_to_improve=3,
        train_ensemble=True)

print(X)
print(y)

automl.fit(X, y)

print(X)
print(y)

predictions = automl.predict(X)
print(predictions.head())
