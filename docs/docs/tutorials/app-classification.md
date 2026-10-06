---
description: Beginner-friendly tutorial for training MLJAR AutoML on the diabetes classification dataset and sharing the model as a Mercury app.
social:
  cards_layout: default/variant
---

# Share a classification model as an app

You trained `AutoML`. What next?

One practical next step is to share the model as an app. With `mljar-supervised`, you can generate a Mercury app from your trained model and let other people use it in a browser.

In this tutorial, we will:

- train a classification model on the diabetes dataset,
- generate an app from the trained model,
- run it locally,
- and optionally publish it online.

## Load data

We will use the diabetes dataset from `datasets-for-start`:

```python
import pandas as pd

data = pd.read_csv(
    "https://raw.githubusercontent.com/pplonski/datasets-for-start/master/diabetes/data.csv"
)

print(data.head())
```

The target column is `Outcome`. All other columns will be used as features.

## Train AutoML

```python
from supervised import AutoML

X = data.drop(columns=["Outcome"])
y = data["Outcome"]

automl = AutoML(
    results_path="AutoML_Diabetes_App",
    mode="Explain",
    random_state=1,
)
automl.fit(X, y)
```

At this point, the model is trained and ready to make predictions.

## Generate the app

Now turn the trained model into an app:

```python
automl.app()
```

This creates an app workspace in:

```text
AutoML_Diabetes_App/app
```

The generated app includes:

- a single prediction form,
- batch CSV scoring,
- configuration files,
- and Python dependencies needed to run the app.

Because this dataset has only `8` input features, the app will include the full single-prediction interface.

## Run the app locally

The easiest way to preview the app is:

```python
automl.local_app()
```

This will:

- generate the app if needed,
- start Mercury,
- open the browser,
- and keep running until you press `Ctrl+C`.

!!! note
    Press `Ctrl+C` in the terminal to stop the local app.

## Run the app manually

You can also start the app yourself. This is useful if you want more control over the environment.

```bash
cd AutoML_Diabetes_App/app
pip install -r requirements.txt
mercury --working-dir=.
```

## What the app looks like

The generated classification app can provide:

- a single prediction form for one patient,
- predicted label and probability summary,
- batch prediction from a CSV file,
- a downloadable scored dataset.

This is a simple way to share your trained model with teammates or stakeholders without asking them to write Python code.

## Publish the app online

If you want the fastest path to sharing the app online, use:

```python
automl.publish_app()
```

This helper will:

- sign you in through `platform.mljar.com`,
- create the app URL on the first publish,
- reuse the last successfully published app URL on later publishes,
- upload the generated app files,
- and print the final app address.

Using `platform.mljar.com` is simply the fastest option.

## Host it yourself

You are not locked into one hosting path.

You can also:

1. generate the app with `automl.app()`,
2. install dependencies on your own server,
3. run Mercury there,
4. and deploy it with your own infrastructure.

## Summary

After training `AutoML`, you do not need to stop at a Python object.

You can turn the trained model into an interactive app and share it:

- locally with `automl.local_app()`,
- online with `automl.publish_app()`,
- or on your own server with manual Mercury startup.
