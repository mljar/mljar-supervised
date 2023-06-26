import numpy as np
import pandas as pd
from supervised.automl import AutoML

df = pd.read_csv("tests/data/LawSchool/bar_pass_prediction.csv")
df["race1"][df["race1"] != "white"] = "non-white"  # keep it as binary feature

X = df[["gender", "lsat", "race1", "pass_bar"]]
y = df["gpa"]

sensitive_features = df["race1"]

automl = AutoML(
    algorithms=["Xgboost", "LightGBM", "Extra Trees"],
    train_ensemble=True,
    fairness_threshold=0.9,
)
automl.fit(X, y, sensitive_features=sensitive_features)
