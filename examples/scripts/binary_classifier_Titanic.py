import pandas as pd
import numpy as np
from supervised.automl import AutoML
import os

from sklearn.metrics import accuracy_score

df = pd.read_csv("tests/data/Titanic/train.csv")

X = df[df.columns[2:]]
y = df["Survived"]

automl = AutoML(mode="Explain")
automl.fit(X, y)
pred = automl.predict(X)

print("Train accuracy", accuracy_score(y, pred))
test = pd.read_csv("tests/data/Titanic/test_with_Survived.csv")
pred = automl.predict(test)
print("Test accuracy", accuracy_score(test["Survived"], pred))
