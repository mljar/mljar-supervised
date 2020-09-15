import pandas as pd
import numpy as np
from supervised.automl import AutoML
import os

from sklearn.metrics import accuracy_score
'''
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
'''

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from supervised import AutoML

train = pd.read_csv("https://raw.githubusercontent.com/pplonski/datasets-for-start/master/Titanic/train.csv")
print(train.head())

X = train[train.columns[2:]]
y = train["Survived"]

automl = AutoML(results_path="AutoML_1") # default mode is Explain
automl.fit(X, y)

test = pd.read_csv("https://raw.githubusercontent.com/pplonski/datasets-for-start/master/Titanic/test_with_Survived.csv")
predictions = automl.predict(test)
print(predictions)
print(f"Accuracy: {accuracy_score(test['Survived'], predictions):.2f}%" )