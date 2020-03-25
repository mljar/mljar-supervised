import pandas as pd
from supervised.automl import AutoML
import os

df = pd.read_csv("https://raw.githubusercontent.com/pplonski/datasets-for-start/master/adult/data.csv", skipinitialspace=True)

X = df[df.columns[:-1]]
y = df["income"]


automl = AutoML(
        total_time_limit=600,
        start_random_models=30,
        hill_climbing_steps=0,
        top_models_to_improve=0,
        train_ensemble=False)

automl.fit(X, y)

predictions = automl.predict(X)
print(predictions.head())
