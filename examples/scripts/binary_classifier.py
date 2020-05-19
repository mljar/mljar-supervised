import pandas as pd
from supervised.automl import AutoML
import os

df = pd.read_csv(
    "https://raw.githubusercontent.com/pplonski/datasets-for-start/master/adult/data.csv",
    skipinitialspace=True,
)

X = df[df.columns[:-1]]
y = df["income"]

automl = AutoML(
    algorithms=["Xgboost"], # ["Linear", "Xgboost", "LightGBM", "Extra Trees"], 
    total_time_limit=600,
    explain_level=0
)
#automl.set_advanced(start_random_models=3)

automl.fit(X, y)
predictions = automl.predict(X)

print(predictions.head())
