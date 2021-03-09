import numpy as np
import pandas as pd
from supervised.automl import AutoML
from sklearn.model_selection import train_test_split
import os
from sklearn.metrics import log_loss
import warnings

# warnings.filterwarnings("error", category=RuntimeWarning) #pd.core.common.SettingWithCopyWarning)

df = pd.read_csv(
    "https://raw.githubusercontent.com/pplonski/datasets-for-start/master/adult/data.csv",
    skipinitialspace=True,
)

X = df[df.columns[:-1]]
y = df["income"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

automl = AutoML(
    algorithms=["LightGBM"],
    mode="Compete",
    explain_level=0,
    train_ensemble=True,
    golden_features=False,
    features_selection=False,
    eval_metric="auc",
)
automl.fit(X_train, y_train)

predictions = automl.predict_all(X_test)

print(predictions.head())
print(predictions.tail())
print(X_test.shape, predictions.shape)
print("LogLoss", log_loss(y_test, predictions["prediction_>50K"]))
