from supervised import AutoML
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.utils.estimator_checks import check_estimator
from sklearn.datasets import load_boston, load_digits
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

# # load the data
# digits = load_digits()
# X_train, X_test, y_train, y_test = train_test_split(
#     pd.DataFrame(digits.data),
#     digits.target,
#     stratify=digits.target,
#     test_size=0.25,
#     random_state=123,
# )
import pandas
import numpy as np

# Load the data
housing = load_boston()

X_train, X_test, y_train, y_test = train_test_split(
    pd.DataFrame(housing.data, columns=housing.feature_names),
    housing.target,
    test_size=0.25,
    random_state=123,
)

# valid_explain_levels = [0, 1, 2]

# return 1
# # X_train = X_train.to_numpy()
# # # values = np.zeros(20)

# # X_train, y_train = check_X_y(X_train, y_train)
# # # print(X_train.head)
# # y_train = pd.DataFrame(y_train, columns=["target"])
# # print(len(y_train))
# # print(X_train.head)
automl = AutoML()
check_estimator(automl)
# automl.fit(X_train, y_train)

# s = pd.Series(list("x1x2x3y"))
# df = pd.read_excel("output_test_multi.xlsx")

# # X, y = make_classification()
# # df = pd.DataFrame({'x': X, 'y': y})
# df = df[
#     [
#         "idade",
#         "Analise Matematica II",
#         "Arquitetura de Computadores",
#         "Ingles II",
#         "Laboratorio Integrado I",
#         "Desistencia",
#     ]
# ]
# X_train, X_test, y_train, y_test = train_test_split(
#     df[df.columns[:-1]], df["Desistencia"], test_size=0.25
# )
# automl.fit(X_train, y_train)
# proba = automl.predict_proba(X_test)
# print(proba)
# # print("....................")
# # print(proba[:, 1])
# # print("....................")

# print(automl.predict(X_test))
# print("....................")

# print(automl._best_model._threshold)
