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
    results_path="AutoML_30",
    #algorithms=["Neural Network"],
    # total_time_limit=200,
    # explain_level=0
    # validation={"validation_type": "split"},
    mode="Compete",
    # validation={"validation_type": "split"}
    validation={
        "validation_type": "kfold",
        "k_folds": 2,
        "shuffle": True,
        "stratify": True,
    }
    # golden_features=False,
    # feature_selection=False
)
automl.set_advanced(
    start_random_models=1, hill_climbing_steps=1, top_models_to_improve=1
)
automl.fit(X_train, y_train)

predictions = automl.predict(X_test)

print(predictions.head())
print(predictions.tail())
print(X_test.shape, predictions.shape)
print("LogLoss", log_loss(y_test, predictions["prediction_>50K"]))
