import pandas as pd
import numpy as np
from supervised.automl import AutoML

# df = pd.read_csv("tests/data/iris_classes_missing_values_missing_target.csv")
df = pd.read_csv("tests/data/iris_missing_values_missing_target.csv")
X = df[["feature_1", "feature_2", "feature_3", "feature_4"]]
y = df["class"]

automl = AutoML(
    results_path="AutoML_100",
    algorithms=["CatBoost"],
    model_time_limit=1, tuning_mode="Normal"
)
automl.fit(X, y)

predictions = automl.predict(X)

print(predictions.head())
print(predictions.tail())
