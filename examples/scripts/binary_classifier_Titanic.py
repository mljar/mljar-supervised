import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from supervised import AutoML

train = pd.read_csv(
    "https://raw.githubusercontent.com/pplonski/datasets-for-start/master/Titanic/train.csv"
)
print(train.head())

X = train[train.columns[2:]]
y = train["Survived"]

automl = AutoML()  # default mode is Explain

automl.fit(X, y)

test = pd.read_csv(
    "https://raw.githubusercontent.com/pplonski/datasets-for-start/master/Titanic/test_with_Survived.csv"
)
predictions = automl.predict(test)
print(predictions)
print(f"Accuracy: {accuracy_score(test['Survived'], predictions)*100.0:.2f}%")
