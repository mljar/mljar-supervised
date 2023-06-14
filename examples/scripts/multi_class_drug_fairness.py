import pandas as pd
import numpy as np

from supervised import AutoML


df = pd.read_csv("tests/data/Drug/Drug_Consumption.csv")


X = df[df.columns[1:13]]

# convert to 3 classes
df = df.replace(
    {
        "Cannabis": {
            "CL0": "never_used",
            "CL1": "not_in_last_year",
            "CL2": "not_in_last_year",
            "CL3": "used_in_last_year",
            "CL4": "used_in_last_year",
            "CL5": "used_in_last_year",
            "CL6": "used_in_last_year",
        }
    }
)

y = df["Cannabis"]

# maybe should be 
# The binary sensitive feature is education level (college degree or not).
# like in 
# Fairness guarantee in multi-class classification
sensitive_features = df["Gender"]


automl = AutoML(
    algorithms=["Xgboost"],
    train_ensemble=False,
    fairness_threshold=0.8
)
automl.fit(X, y, sensitive_features=sensitive_features)
