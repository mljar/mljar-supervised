import pandas as pd
import numpy as np
from supervised.automl import AutoML
import supervised 

print(supervised.__version__)
import sys
print(sys.path)


# df = pd.read_csv("tests/data/iris_classes_missing_values_missing_target.csv")
df = pd.read_csv("tests/data/iris_missing_values_missing_target.csv")
X = df[["feature_1", "feature_2", "feature_3", "feature_4"]]
y = df["class"]

automl = AutoML(
    results_path="AutoML_41",
    algorithms=["LightGBM"],
    #algorithms=["Random Forest"],
    #    "Linear",
    #    "Xgboost",
    #    "Random Forest"
    #],
    model_time_limit=10111,
    tuning_mode="Normal",
    explain_level=0
)
automl.set_advanced(start_random_models=1)
automl.fit(X, y)

predictions = automl.predict(X)

print(predictions.head())
print(predictions.tail())
