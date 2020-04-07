import pandas as pd
from supervised.automl import AutoML
import os

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

df = pd.read_csv("tests/data/PortugeseBankMarketing/Data_FinalProject.csv")

X = df[df.columns[:-1]]
y = df["y"]


X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25)


automl = AutoML(
    # results_path="AutoML_22",
    total_time_limit=30 * 60,
    start_random_models=10,
    hill_climbing_steps=3,
    top_models_to_improve=3,
    train_ensemble=True,
)

automl.fit(X_train, y_train)


pred = automl.predict(X_test)
print("Test accuracy", accuracy_score(y_test, pred["label"]))
