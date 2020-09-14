import pandas as pd
import numpy as np
from supervised.automl import AutoML
import supervised


import warnings

from sklearn import datasets
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA

from supervised import AutoML
from supervised.exceptions import AutoMLException

# warnings.filterwarnings('error')
warnings.filterwarnings(
    "error", category=pd.core.common.SettingWithCopyWarning
)  # message="*ndarray*")

df = pd.read_csv("tests/data/iris_missing_values_missing_target.csv")
X = df[["feature_1", "feature_2", "feature_3", "feature_4"]]
y = df["class"]

automl = AutoML()

automl.fit(X, y)

predictions = automl.predict_all(X)

print(predictions.head())
print(predictions.tail())

print(X.shape)
print(predictions.shape)
