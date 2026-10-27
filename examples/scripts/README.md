# Follow a tutorial with a Python script

Each new beginner tutorial has its own standalone script. Download it from the
tutorial page or run it from this directory. You do not need the repository's
notebook generator, screenshot tools, or MkDocs to train a model.

| Tutorial | Script | What you will do |
| --- | --- | --- |
| [Predict a number from a CSV](https://supervised.mljar.com/tutorials/concrete-strength/) | [concrete_strength.py](concrete_strength.py) | Predict concrete strength, compare held-out errors, read reports, and reload the model. |
| [Predict which customers may leave](https://supervised.mljar.com/tutorials/customer-churn/) | [customer_churn.py](customer_churn.py) | Predict churn probabilities, inspect classification errors, and reload the saved model. |

Use Python 3.11 or 3.12 in a virtual environment. For the concrete example:

```bash
python -m pip install 'mljar-supervised==1.4.0b1'
python concrete_strength.py --output AutoML_ConcreteStrength
```

Read the numbered steps in `run()` from top to bottom. The script downloads its
reviewed dataset, prints training progress and test scores, and writes predictions,
a chart, and an HTML report. Open `AutoML_ConcreteStrength/automl/README.html` in
your browser. Keep the entire `automl` directory to reuse the saved model.

Use a different output directory to train again. Run `python concrete_strength.py
--help` for options. The default training budget is 120 seconds with two CPU
threads; total runtime includes other work and varies by machine.

Other scripts in this directory are existing examples with their own dependencies
and assumptions. They have not all been converted to this beginner format.

For churn, use the same installation and run
`python customer_churn.py --output AutoML_CustomerChurn`. Read the six numbered
steps, then open `AutoML_CustomerChurn/automl/README.html`.

For maintainers: edit the tutorial's script as the source of its executable code.
Keep documentation generation separate; see [the notebook maintenance guide](../notebooks/README.md).

## Custom-validation recipes

These standalone scripts use small synthetic regression data to explain splits;
they are not real-domain performance benchmarks. Both use the same beta install,
a 60-second training budget, two CPU threads, and no downloads.

```bash
python custom_validation_grouped.py --output AutoML_GroupedValidation
python custom_validation_time.py --output AutoML_TimeValidation
```

Read [the custom-validation guide](../../docs/docs/features/custom-validation.md)
for group separation, chronological windows, gaps, and saved-fold behavior.
Each run saves the input data, split indices and diagram, test predictions,
an HTML report, and a fresh-process reload comparison.
