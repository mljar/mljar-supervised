<p><img src="docs/docs/images/brand/mljar-logo.webp" alt="MLJAR" width="64" /></p>

# Build machine learning models from your tabular data

MLJAR AutoML trains and compares models, creates reports, and saves the results
for future predictions. Use it for classification and regression with data from
CSV files, spreadsheets, or pandas DataFrames.

[Documentation](https://supervised.mljar.com/) ·
[Example gallery](docs/docs/examples.md) ·
[Installation](docs/docs/install.md) ·
[Release notes](release.md)

## Choose your starting point

| Python | MLJAR Studio |
| --- | --- |
| Follow a standalone script or notebook, from loading data to evaluating and reloading a model. | Work in a conversational notebook with visible Python code. |
| [Start with a Python tutorial](docs/docs/tutorials/concrete-strength.md) | [Explore MLJAR Studio](https://mljar.com/studio/) |

The two beginner workflows below were tested as Python scripts and notebooks.
Their Studio walkthroughs are still being validated.

## Quickstart

Use Python 3.11 or 3.12 in a virtual environment. Install the exact beta used by
the new tutorials:

```bash
python -m pip install 'mljar-supervised==1.4.0b1'
```

Save [customer_churn.py](examples/scripts/customer_churn.py) in your working
folder, then run:

```bash
python customer_churn.py --output AutoML_CustomerChurn
```

The script downloads a reviewed public dataset, compares standard CPU models
against a baseline, evaluates held-out customers, and verifies predictions after
reload. It uses a 120-second training budget and two CPU threads; total runtime
varies. No GPU or optional Torch packages are needed.

Open `AutoML_CustomerChurn/automl/README.html` to read your report. In a new
Python session, predict with the saved model:

```python
import pandas as pd
from supervised import AutoML

model = AutoML(results_path="AutoML_CustomerChurn/automl")
customers = pd.read_csv("AutoML_CustomerChurn/new_customers.csv")
churn_probability = model.predict_proba(customers)[:, 1]
print(churn_probability)
```

The example target is numeric: 0 means stayed, 1 means churned. The CSV contains
three held-out example inputs, not independently collected new customers. Keep
the complete `automl` directory for future predictions. Use a new output path to
train again. [Follow the full tutorial](docs/docs/tutorials/customer-churn.md)
for the data, validation choices, and interpretation.

## Examples

Start with a question similar to yours. Each tutorial includes a standalone
script, a notebook, dataset attribution, measured results, and real reports.

| Predict a number from a CSV | Predict which customers may leave |
| --- | --- |
| [![Measured versus predicted concrete strength on held-out mixture groups.](docs/docs/assets/concrete-strength/predicted-vs-measured.png)](docs/docs/tutorials/concrete-strength.md) | [![Baseline and AutoML confusion matrices for held-out customers at threshold 0.5.](docs/docs/assets/customer-churn/confusion-matrix.png)](docs/docs/tutorials/customer-churn.md) |
| **Start here · Engineering · Regression** | **Start here · Business · Classification** |
| Estimate concrete strength from ingredients and age. Learn to measure error in MPa and keep related measurements together. | Estimate churn probabilities from service usage. Understand missed customers, false alarms, and a fixed decision threshold. |
| [Read the tutorial](docs/docs/tutorials/concrete-strength.md) · [Script](examples/scripts/concrete_strength.py) · [Notebook](examples/notebooks/concrete_strength.ipynb) | [Read the tutorial](docs/docs/tutorials/customer-churn.md) · [Script](examples/scripts/customer_churn.py) · [Notebook](examples/notebooks/customer_churn.ipynb) |

Both scripts and notebooks passed on Python 3.11 and 3.12 with published
`mljar-supervised==1.4.0b1`. Their tutorials record the environments, evaluation
protocols, runtime, memory measurements, and reload checks. Results from one
historical split are not promises about another dataset or machine.

[Browse the gallery and advanced guides](docs/docs/examples.md).

<a id="automated-machine-learning"></a>
<a id="whats-good-in-it"></a>

## What you can do with AutoML

| Capability | What it provides | Learn more |
| --- | --- | --- |
| Train and compare models | Preprocessing, model selection, hyperparameter search, and ensembles | [Algorithms](docs/docs/features/algorithms.md), [training stages](docs/docs/features/automl.md) |
| Generate features | Golden features and feature discovery, evaluated through validation | [Golden features](docs/docs/features/golden_features.md), [discovery](docs/docs/features/feature_discovery.md) |
| Try tabular foundation models | Optional Kumo and TabICLv2 backends | [TFM setup and requirements](docs/docs/features/sdm.md) |
| Try neural tabular models | Optional RealMLP and TabM backends | [RealMLP](docs/docs/features/realmlp.md), [TabM](docs/docs/features/tabm.md) |
| Read automatic reports | Run and model reports, metrics, settings, and available explanations | [Reports](docs/docs/features/reports.md), [explainability](docs/docs/features/explain.md) |
| Evaluate fairness and bias mitigation | Group metrics and mitigation evaluated alongside predictive performance | [Fairness tutorial](docs/docs/tutorials/fairness-binary-classification.md) |
| Export a prediction app | Interfaces for individual rows and batch CSV predictions | [App guide](docs/docs/features/apps.md) |
| Customize evaluation | Custom metrics and user-provided validation splits | [Custom metrics](docs/docs/features/custom-eval-metric.md), [custom validation](docs/docs/features/custom-validation.md) |

Feature discovery is included in the base installation. Its use depends on the
mode and validation setup; the beginner examples disable it for custom grouped
validation. Optional models have additional dependencies and resource limits.
Explanations depend on the algorithm and `explain_level`.

## Automatic documentation

Every successful run provides a leaderboard and links to individual model
reports. Inspect the validation scores, model settings, and available charts.
The beginner tutorials show real run and model screenshots and explain how to
read them.

Reports are also available as structured Markdown, dictionaries, or JSON through
`report_structured()`. [Read the reports guide](docs/docs/features/reports.md).

## Available modes

Choose `Explain`, `Perform`, `Compete`, or `Optuna` according to the workflow and
compute budget. Modes configure search, validation, ensembling, and explanations;
you can override individual settings. See [modes](docs/docs/features/modes.md),
[training stages](docs/docs/features/automl.md), and the [API reference](docs/docs/api.md).

## Fairness aware training

Measure relevant group differences and evaluate bias mitigation together with
predictive performance. Choosing the groups, metrics, and acceptable trade-offs
requires context; one score does not establish universal fairness.

Start with [binary classification](docs/docs/tutorials/fairness-binary-classification.md),
[multiclass classification](docs/docs/tutorials/fairness-multiclass-classification.md),
or [regression](docs/docs/tutorials/fairness-regression.md). Check backend support
for sample weights before combining mitigation with optional algorithms.

## Generate web apps for trained models

The base package exports app files with `automl.app()`. Running an exported app
requires the optional Mercury runtime and the selected model's dependencies.
Generation, local execution, and publishing are separate steps.

See the [app guide](docs/docs/features/apps.md) for single-row and batch CSV
prediction, saved artifacts, and deployment options.

## Installation

Version 1.4.0b1 requires Python 3.11+. The beginner workflows were tested on
Python 3.11 and 3.12. Use the exact beta pin above to reproduce their environment;
an unpinned install may select a different release.

| Install option | Additional capabilities |
| --- | --- |
| `mljar-supervised==1.4.0b1` | Standard algorithms, reports, and feature discovery |
| `mljar-supervised[realmlp]==1.4.0b1` or `[tabm]` | Shared RealMLP/TabM backend dependencies |
| `mljar-supervised[tfm]==1.4.0b1` | Kumo/TabICLv2 backend; weights download on first use |
| `mljar-supervised[apps]==1.4.0b1` | Runtime for generated prediction apps |
| `mljar-supervised[all]==1.4.0b1` | All optional model backends |
| `mljar-supervised[all,apps]==1.4.0b1` | Optional model backends and app runtime |

For Torch-based extras, choose a compatible CPU or CUDA PyTorch build first.
See the [installation guide](docs/docs/install.md) for exact commands, aliases,
platform limitations, and native-runtime compatibility checks. The base package
does not require Torch or Mercury. Conda-forge releases are maintained separately.

## FAQ

- **How do I save and reload?** Models are saved in `results_path`. Keep the
  complete directory; see [save and load](docs/docs/features/save-and-load-models.md).
- **Can I choose classification or regression explicitly?** Set `ml_task`.
  See the [API reference](docs/docs/api.md) for supported values.
- **How do I identify probability columns?** Use `predict_all()` to inspect the
  class-labelled prediction columns. Their order matches `predict_proba()`;
  `predict_all()` also includes the predicted label.
- **How is tuning configured?** See [AutoML stages](docs/docs/features/automl.md)
  and the [API reference](docs/docs/api.md), including `optuna_init_params` for
  reusing Optuna parameters.

## Documentation

[Documentation website](https://supervised.mljar.com/) ·
[Example gallery](docs/docs/examples.md) · [API reference](docs/docs/api.md) ·
[Get help](docs/docs/get_help.md) · [Report an issue](https://github.com/mljar/mljar-supervised/issues)

<a id="demo"></a>

For a walkthrough with real outputs, follow the
[concrete-strength tutorial](docs/docs/tutorials/concrete-strength.md) or
[customer-churn tutorial](docs/docs/tutorials/customer-churn.md).

## Contributing

Read the [contribution guide](docs/docs/contributing.md) and
[test instructions](tests/README.md). Thanks to everyone who has
[contributed](https://github.com/mljar/mljar-supervised/graphs/contributors).

## Cite

The project's published citation is:

```bibtex
@misc{mljar,
  author    = {Aleksandra P\l{}o\'{n}ska and Piotr P\l{}o\'{n}ski},
  year      = {2021},
  publisher = {MLJAR},
  address   = {\L{}apy, Poland},
  title     = {MLJAR: State-of-the-art Automated Machine Learning Framework for Tabular Data.  Version 0.10.3},
  url       = {https://github.com/mljar/mljar-supervised}
}
```

## License

MLJAR AutoML is available under the [MIT license](LICENSE). Example datasets and
optional model weights have their own licenses, documented with their sources.

## Commercial support

For implementation or commercial support, [contact MLJAR](https://mljar.com/contact/).

## MLJAR

Created by [MLJAR](https://mljar.com/). Explore
[MLJAR Studio](https://mljar.com/studio/) for a conversational notebook interface.
