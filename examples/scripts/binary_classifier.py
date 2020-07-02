import numpy as np
import pandas as pd
from supervised.automl import AutoML
from sklearn.model_selection import train_test_split
import os

import warnings
#warnings.filterwarnings("error", category=pd.core.common.SettingWithCopyWarning) 

df = pd.read_csv(
    "https://raw.githubusercontent.com/pplonski/datasets-for-start/master/adult/data.csv",
    skipinitialspace=True,
)

X = df[df.columns[:-1]]
y = df["income"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

automl = AutoML(
    results_path="AutoML_8",
    algorithms=["Baseline", "Decision Tree", "Xgboost", "CatBoost"], 
    total_time_limit=10*60,
    explain_level=0
)
automl.fit(X_train, y_train)

predictions = automl.predict(X_test)

print(predictions.head())
print(predictions.tail())
print(X_test.shape, predictions.shape)
