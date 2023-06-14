import numpy as np
import pandas as pd
from supervised.automl import AutoML

# to get data
# from fairlearn.datasets import fetch_acs_income
# df = fetch_acs_income(as_frame=True)
# df["frame"].to_csv("acs_income.csv", index=False)

df = pd.read_csv("tests/data/acs_income_1k.csv")

print(df)

x_cols = [c for c in df.columns if c != "PINCP"]

sensitive_features = df["SEX"].astype(str)

X = df[x_cols]
y = df["PINCP"]

automl = AutoML(
    algorithms=["Xgboost", "LightGBM"],
    train_ensemble=True,
    fairness_threshold=0.91,
    # underprivileged_groups=[{"SEX": "1.0"}],
    # privileged_groups=[{"SEX": "2.0"}]
)
automl.fit(X, y, sensitive_features=sensitive_features)
