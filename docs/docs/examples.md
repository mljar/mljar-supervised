---
title: Examples and guides
description: Learn MLJAR AutoML with tested regression and classification tutorials, then explore custom metrics, validation, feature engineering, fairness, and prediction apps.
---

# Find an example for your data

Start with a question similar to yours. Each beginner tutorial includes the data
source, a standalone Python script, a notebook, measured results, and real reports.
Both use standard CPU models and a 120-second training budget. Total runtime
varies; each tutorial records its measured time and hardware.

## Start here

<div class="mljar-cards mljar-cards--two" markdown="1">
<div class="mljar-card mljar-example" markdown="1">

[![Measured and predicted concrete strength from a recorded held-out evaluation.](assets/concrete-strength/predicted-vs-measured.png)](tutorials/concrete-strength.md)

<p class="mljar-eyebrow">Engineering · Regression</p>

### :material-office-building-outline: Predict a number from a CSV

**Question:** can ingredients and specimen age predict concrete strength?

Learn to choose a target, keep related measurements together, compare errors
with a baseline, and reuse a saved model. Data: 1,030 measurements, eight inputs.

[Read the tutorial](tutorials/concrete-strength.md){ .md-button .md-button--primary }

[Python script](assets/concrete-strength/concrete_strength.py){ download="concrete_strength.py" } ·
[Notebook](assets/concrete-strength/concrete_strength.ipynb){ download="concrete_strength.ipynb" } ·
[Full report](assets/concrete-strength/report.html)

</div>
<div class="mljar-card mljar-example" markdown="1">

[![Baseline and AutoML confusion matrices at a fixed threshold on held-out customers.](assets/customer-churn/confusion-matrix.png)](tutorials/customer-churn.md)

<p class="mljar-eyebrow">Business · Classification</p>

### :material-account-arrow-right-outline: Predict which customers may leave

**Question:** can past service usage help identify customers likely to churn?

Learn probabilities, class imbalance, missed churn, and false alarms. Keep
repeated profiles together and verify saved-model predictions. Data: 3,150 rows,
ten selected inputs.

[Read the tutorial](tutorials/customer-churn.md){ .md-button .md-button--primary }

[Python script](assets/customer-churn/customer_churn.py){ download="customer_churn.py" } ·
[Notebook](assets/customer-churn/customer_churn.ipynb){ download="customer_churn.ipynb" } ·
[Full report](assets/customer-churn/report.html)

</div>
</div>

These workflows passed script and notebook execution on Python 3.11 and 3.12
with published MLJAR AutoML 1.4.0b1. Their pages link to dataset attribution,
recorded versions, resource measurements, and reload checks. Studio walkthroughs,
laptop/8 GB measurements, and unassisted beginner reviews remain open.

## Choose by learning goal

| I want to… | Start with |
| --- | --- |
| Predict a number, such as a measurement | [Concrete strength](tutorials/concrete-strength.md) |
| Predict a category and understand probabilities | [Customer churn](tutorials/customer-churn.md) |
| Understand the leaderboard and individual reports | [Read the reports](tutorials/concrete-strength.md#read-the-reports) |
| Make predictions after closing Python | [Predict another CSV and reload](tutorials/customer-churn.md#predict-another-csv-and-reload) |
| Use a conversational notebook | [MLJAR Studio](https://mljar.com/studio/) and a beginner notebook above |

## Advanced guides

Once you have a first model, change one part of the workflow at a time. These
links include existing reference guides and tutorials; they are not all newly
validated 1.4.0b1 example packages. Check each page's dependencies and provenance.

| Task | Guide | What to check |
| --- | --- | --- |
| Define your own evaluation metric | [Custom metrics](features/custom-eval-metric.md) | Optimization direction and the form of the predictions |
| Supply your own validation splits | [Custom validation](features/custom-validation.md) | Tested grouped and time-ordered scripts; separate final test data, gaps, and fold-trained models |
| Generate additional features | [Golden features](features/golden_features.md) and [feature discovery](features/feature_discovery.md) | Validation compatibility, compute cost, and measured benefit |
| Understand model explanations | [Explainability](features/explain.md) | Which explanations are available for your algorithm and explanation level |
| Read reports in Python or an LLM workflow | [Reports and structured output](features/reports.md) | Validation versus test scores and available model details |
| Measure fairness and evaluate mitigation | [Binary classification](tutorials/fairness-binary-classification.md), [multiclass](tutorials/fairness-multiclass-classification.md), [regression](tutorials/fairness-regression.md) | Relevant groups and metrics; performance trade-offs |
| Share single-row and batch predictions | [Prediction apps](features/apps.md) | Runtime dependencies, selected-model artifacts, and local execution versus publishing |
| Try neural tabular models | [RealMLP](features/realmlp.md) and [TabM](features/tabm.md) | Optional backend installation and resource requirements |
| Try tabular foundation models | [Kumo and TabICLv2](features/sdm.md) | Compatible hardware, first-use model downloads, and saved artifacts |
| Choose a search strategy | [AutoML modes](features/modes.md) and [training stages](features/automl.md) | Budget, validation, and the work each mode performs |

## More existing examples

These entry points remain available while the collection is reviewed and refreshed.
Their historical results should not be read as new beta measurements.

- [Classify Titanic passengers](tutorials/titanic.md).
- [Train with IP addresses](tutorials/ip-addresses.md).
- [Classify synthetic data](tutorials/random.md) for a focused API example.
- [Share a classification model as an app](tutorials/app-classification.md).
- [Share a regression model as an app](tutorials/app-regression.md).

Need a setting rather than a tutorial? Open the [API reference](api.md) or
[installation guide](install.md).
