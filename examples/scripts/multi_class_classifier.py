import pandas as pd
import numpy as np
from supervised.automl import AutoML
import supervised


import warnings

# warnings.filterwarnings('error')
warnings.filterwarnings(
    "error", category=pd.core.common.SettingWithCopyWarning
)  # message="*ndarray*")


# df = pd.read_csv("tests/data/iris_classes_missing_values_missing_target.csv")
df = pd.read_csv("tests/data/iris_missing_values_missing_target.csv")
X = df[["feature_1", "feature_2", "feature_3", "feature_4"]]
y = df["class"]

automl = AutoML(
    # results_path="AutoML_41",
    #algorithms=["Random Forest"],
    # algorithms=["Neural Network"],
    #    "Linear",
    #    "Xgboost",
    #    "Random Forest"
    # ],
    # total_time_limit=100,
    # tuning_mode="Normal",
    # explain_level=0,
    #mode="Perform"
)
# automl.set_advanced(start_random_models=1)
automl.fit(X, y)

predictions = automl.predict(X)

print(predictions.head())
print(predictions.tail())

print(X.shape)
print(predictions.shape)
