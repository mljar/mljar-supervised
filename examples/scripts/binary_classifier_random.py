import numpy as np
import pandas as pd
from supervised.automl import AutoML
from sklearn.metrics import accuracy_score
import os

nrows = 100
ncols = 3
X = np.random.rand(nrows, ncols)
X = pd.DataFrame(X, columns=[f"f{i}" for i in range(ncols)])
#y = np.random.randint(0, 2, nrows)
y = np.random.permutation(["a", "B"] * 50) 

automl = AutoML(model_time_limit=10) #, algorithms=["Decision Tree"])
automl.fit(X, y)
print("Train accuracy", accuracy_score(y, automl.predict_all(X)["label"]))

#X = np.random.rand(1000, 10)
#X = pd.DataFrame(X, columns=[f"f{i}" for i in range(10)])
#y = np.random.randint(0, 2, 1000)
#print("Test accuracy", accuracy_score(y, automl.predict(X)["label"]))
