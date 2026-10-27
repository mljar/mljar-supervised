# Read and reuse AutoML reports

AutoML saves a run-level `README.md` and individual model reports in its results
directory. The leaderboard compares validation scores. Keep final held-out test
results separate from the scores used to select models.

## Open the HTML report

After running the [customer-churn tutorial](../tutorials/customer-churn.md), load
the saved run and create the HTML report:

```python
from supervised import AutoML

model = AutoML(results_path="AutoML_CustomerChurn/automl")
model.report()
```

Open `AutoML_CustomerChurn/automl/README.html` in a browser. In a notebook, leave
`model.report()` as the last expression in a cell to display it. Click a model
name to inspect its settings, validation metrics, and available explanations.

The [concrete-strength walkthrough](../tutorials/concrete-strength.md#read-the-reports)
and [churn walkthrough](../tutorials/customer-churn.md#read-the-reports) show actual
reports from recorded beta runs. For classification, check the threshold next to
each metric: scores shown at different thresholds need not occur together.

## Use a structured report

`report_structured()` provides text or data for analysis tools and LLM workflows.
Using the loaded `model` above:

```python
markdown_report = model.report_structured()
payload = model.report_structured(format="dict")
payload_json = model.report_structured(format="json")
```

Each call creates or updates `report_structured.json` in the results directory.
The default output is Markdown text. For one model, pass its exact saved name:

```python
detailed = model.report_structured(model_name="3_Linear")
print(detailed)
```

`3_Linear` exists in the recorded churn tutorial; use a name from your own
leaderboard for another run. A selected-model report includes detailed metrics,
available feature-importance summaries, and fairness details when present.

## Inspect explanations

Available plots depend on the algorithm and `explain_level`. See
[explainability](explain.md) for learning curves, permutation importance, SHAP,
and model blueprints. Importance describes model reliance on an input; it does
not establish that changing the input causes an outcome to change.

## Keep the saved model artifacts

Reports help explain the result, but they are not a substitute for the model
files. Preserve the complete results directory for prediction and reload,
including preprocessing and ensemble dependencies. See
[saving and loading](save-and-load-models.md).
