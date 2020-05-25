import numpy as np
import pandas as pd
from supervised.automl import AutoML


df = pd.read_csv("./tests/data/housing_regression_missing_values_missing_target.csv")
x_cols = [c for c in df.columns if c != "MEDV"]
X = df[x_cols]
y = df["MEDV"]

print("y", y[:10])

print(X.shape)

automl = AutoML(
    #results_path="AutoML_56",
    model_time_limit=600,
    algorithms=["Neural Network"],
    #    "Xgboost",
        # "Xgboost", "Decision Tree",
        # "Random Forest",
        # "CatBoost",
        # "LightGBM", "Extra Trees"
    #],
    explain_level=0,
    tuning_mode="Normal"

)
#automl.set_advanced(start_random_models=3)
automl.fit(X, y)

df["predictions"] = automl.predict(X)
print("Predictions")
print(df[["MEDV", "predictions"]].head())
