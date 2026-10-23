# Model blueprint examples

Set `explain_level=2` to embed fitted model blueprints in AutoML model READMEs.

These six diagrams use illustrative fitted metadata. Integration tests also train real models, including ensembles and stacking.

Every SVG includes a clickable ‘Trained with MLJAR AutoML’ footer in standalone SVGs and inline HTML reports.

Regenerate with `venv/bin/python -m examples.blueprints.generate`.

## Regression

![Regression_Linear](regression.svg)

## Multiclass Cv

![Multiclass_RandomForest_CV](multiclass_cv.svg)

## Binary Ensemble

![Binary_Ensemble](binary_ensemble.svg)

## Stacked Regression

![Regression_Stacked](stacked_regression.svg)

## Mixed Features

![Binary_Text_IP_Datetime](mixed_features.svg)

## Mixed Ensemble

![Mixed_Stacked_Ensemble](mixed_ensemble.svg)
