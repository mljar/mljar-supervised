import pandas as pd
from supervised.automl import AutoML
import os

from sklearn.metrics import accuracy_score

df = pd.read_csv("tests/data/Titanic/train.csv")

X = df[df.columns[2:]]
y = df["Survived"]

automl = AutoML(
    #results_path="AutoML_1",
    #algorithms=["Xgboost"],
    # model_time_limit=20,
    # train_ensemble=True,
    mode="Perform"
)
# automl.set_advanced(start_random_models=3)
automl.fit(X, y)

pred = automl.predict(X)

print("Train accuracy", accuracy_score(y, pred["label"]))

test = pd.read_csv("tests/data/Titanic/test_with_Survived.csv")
test_cols = [
    "Parch",
    "Ticket",
    "Fare",
    "Pclass",
    "Name",
    "Sex",
    "Age",
    "SibSp",
    "Cabin",
    "Embarked",
]
pred = automl.predict(test[test_cols])
print("Test accuracy", accuracy_score(test["Survived"], pred["label"]))
