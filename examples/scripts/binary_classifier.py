import pandas as pd
from supervised.automl import AutoML
import os

df = pd.read_csv(
    "https://raw.githubusercontent.com/pplonski/datasets-for-start/master/adult/data.csv",
    skipinitialspace=True,
)

X = df[df.columns[:-1]]
y = df["income"]

automl = AutoML(algorithms=["Linear", "Xgboost", "LightGBM", "Extra Trees"], model_time_limit=1)
automl.set_advanced(start_random_models=1)

automl.fit(X, y)
predictions = automl.predict(X)

print(predictions.head())
