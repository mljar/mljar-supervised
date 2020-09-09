import pandas as pd
import numpy as np
from supervised.automl import AutoML
import os

from sklearn.metrics import accuracy_score

"""
obj_array = np.array([1, 2, "A"], dtype=object)
y = pd.DataFrame(obj_array)
X = y.copy()

print(X)
print(np.unique(y[~pd.isnull(y)]))
for col in X.columns:
    print(col, X[col].dtype)

a = AutoML(total_time=30)

a.fit(X, y)
"""

df = pd.read_csv("tests/data/Titanic/train.csv")

X = df[df.columns[2:]]
y = df["Survived"]

automl = AutoML(mode="Perform", explain_level=2, total_time_limit=1*6, feature_selection=True)
automl.fit(X, y)
pred = automl.predict(X)

print("Train accuracy", accuracy_score(y, pred)) #["label"]))

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
pred = automl.predict_all(test[test_cols])
print("Test accuracy", accuracy_score(test["Survived"], pred["label"]))
