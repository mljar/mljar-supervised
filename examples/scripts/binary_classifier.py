import numpy as np
import pandas as pd
from supervised.automl import AutoML
import os

import warnings
#warnings.filterwarnings('error')
warnings.filterwarnings("error", category=pd.core.common.SettingWithCopyWarning) # message="*ndarray*")

df = pd.read_csv(
    "https://raw.githubusercontent.com/pplonski/datasets-for-start/master/adult/data.csv",
    skipinitialspace=True,
)

X = df[df.columns[:-1]]
y = df["income"]

automl = AutoML(
    #results_path="AutoML_23",
    #algorithms=["Baseline", "Decision Tree"], 
    total_time_limit=1*60,
    #explain_level=0
)
#automl.set_advanced(start_random_models=1)
#with warnings.catch_warnings(record=True) as warns:
#    warnings.simplefilter("always")
automl.fit(X, y)


#for w in warns:
#    print(w.category.__name__, "(%s)" % w.message)
#    print( "in", w.filename, "at line", w.lineno)

predictions = automl.predict(X)

print(predictions.head())
