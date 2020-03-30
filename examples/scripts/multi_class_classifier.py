import pandas as pd
import numpy as np
from supervised.automl import AutoML

'''
from sklearn import metrics

# Constants
C="Cat"
F="Fish"
H="Hen"
D="Dupa"

# True values
y_true = [C,C,C,C,C,C, F,F,F,F,F,F,F,F,F,F, H,H,H,H,H,H,H,H,H, D,D,D]
# Predicted values
y_pred = [C,C,C,C,H,F, C,C,C,C,C,C,H,H,F,F, C,C,C,H,H,H,H,H,H, D,D,D]

# Print the confusion matrix
conf_matrix = metrics.confusion_matrix(y_true, y_pred, labels=[C,F,H,D])
print(conf_matrix)

rows = [f"Predicted as {a}" for a in [C,F,H,D]]
cols = [f"Labeled as {a}" for a in [C,F,H,D]]

conf_matrix = pd.DataFrame(
            conf_matrix,
            columns=rows,
            index=cols,
        )

print(conf_matrix)
# Print the precision and recall, among other metrics
m = metrics.classification_report(y_true, y_pred, digits=3, labels= [C,F,H,D], output_dict=True)


print(m)
print(pd.DataFrame( m).transpose())
'''

df = pd.read_csv("tests/data/iris_missing_values_missing_target.csv")
X = df[["feature_1","feature_2","feature_3","feature_4"]]
y = df["class"]

print(X)
print(y)

automl = AutoML(total_time_limit=10)
automl.fit(X, y)

predictions = automl.predict(X)

print(predictions.head())
print(predictions.tail())
