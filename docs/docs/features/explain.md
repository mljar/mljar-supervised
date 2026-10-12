# Explainability in AutoML

There are three modes of explanations available in [`mljar-supervised`](https://github.com/mljar/mljar-supervised). The explanations are controlled by `explain_level` parameter in `AutoML` constructor.

- if `explain_level` is `0` no explanations are produced. Only learning curves are plotted.
- if `explain_level` is `1` the following explanations are produced: learning curves, importance plot (with permutation method), for decision trees produce tree plots, for linear models save coefficients.
- if `explain_level` is `2` the following explanations are produced: the same as `1` plus SHAP explanations and model blueprints.

## Model blueprints

Set `explain_level=2` to include a prediction pipeline diagram in the README of
every successfully saved model, including Baseline, ensembles, and stacked models.

```python
from supervised import AutoML

automl = AutoML(results_path="AutoML_results", explain_level=2)
automl.fit(X, y)
automl.report()
```

Each model directory contains `blueprint.svg`, the diagram, and `blueprint.json`,
its graph representation. The model's `README.md` embeds the SVG, and the main
AutoML README links to all model reports. The HTML report displays the same diagrams.
No additional packages are needed for blueprint generation.

Blueprints use saved fitted preprocessing and learner metadata. They show the
transformation order, learner parameters, prediction averaging, ensemble members
and normalized weights, and stacking dependencies. Matching cross-validation fold
pipelines are grouped, while different transformations remain separate branches.
Each fold retains its own fitted preprocessing and learner. Stacking uses
out-of-fold predictions for training and fitted base-model predictions at inference.

The clickable SVG footer reads **Trained with MLJAR AutoML**. Generation does not
load the training dataset or estimator. Successfully saved models keep their
diagrams if later training is interrupted. If diagram generation fails, AutoML
issues a warning and continues training. Blueprint generation is disabled at
`explain_level=0` and `1`.

This example uses illustrative metadata to show five multiclass cross-validation
learners, with one using a different categorical encoding. Real AutoML blueprints
are generated from the fitted models in your run.

![Multiclass cross-validation blueprint](../images/blueprints/multiclass_cv.svg)

More illustrative diagrams and a regeneration script are available in
[the example gallery](https://github.com/mljar/mljar-supervised/tree/master/examples/blueprints).

## Learning curves

The learning curves show the evaluation metric values in each iteration of the training. The learning curves are plotted for training and validation datasets. The vertical line is used to show the optimal iteration number, which will be later used for computing predictions. Learning curves are always created.

![Learning curves for Xgboost](https://raw.githubusercontent.com/mljar/mljar-examples/master/Titanic_Classification/AutoML_3/5_Default_Xgboost/learning_curves.png)

## `Decision Tree` Visualization

The visualization of the `Decision Tree` is created if `explain_level >= 1`. The [`dtreeviz`](https://github.com/parrt/dtreeviz) is used to plot the tree.

![Decision Tree visualization](https://raw.githubusercontent.com/mljar/mljar-examples/1295c77b6ac617b8d91ea7d8fffc6cd4c2605701/Income_classification/AutoML_1/2_DecisionTree/learner_1_tree.svg)

## `Linear` model coefficients

For the `explain_level >= 1` the coefficents of the `Linear` model are saved in the Markdown report. The example of cofficents is presented below.

| feature        |   Learner_1 |
|:---------------|------------:|
| capital-gain   |  2.28219    |
| education-num  |  0.843846   |
| age            |  0.468591   |
| sex            |  0.468299   |
| hours-per-week |  0.369091   |
| capital-loss   |  0.279562   |
| race           |  0.104163   |
| education      |  0.0546121  |
| fnlwgt         |  0.0545988  |
| native-country |  0.0173909  |
| occupation     | -0.00958272 |
| workclass      | -0.102386   |
| relationship   | -0.154081   |
| marital-status | -0.358737   |
| intercept      | -1.51172    |

## Features Importance

The features importance is computed with permutation-based method (using scikit-learn [`permutation_importance`](https://scikit-learn.org/stable/modules/generated/sklearn.inspection.permutation_importance.html)). It is available for algorithms other than `Baseline`, subject to the kNN size limit below. The importance is presented in the plot (top-25 importance features) and saved to the file `learner_*_importance.csv` for all features. It needs `explain_level >= 1`.

![Permutation Importance](https://raw.githubusercontent.com/mljar/mljar-examples/master/Income_classification/AutoML_1/5_Default_Xgboost/permutation_importance.png)

### kNN interpretation size limit

For `Nearest Neighbors` classification and regression, interpretation is skipped
when either the learner's training fold or validation fold has **more than
10,000 rows**. This avoids expensive repeated predictions for permutation
importance on large datasets. The check uses the full fold sizes, before the
internal kNN fitting subsample; it applies at both `explain_level=1` and `2`.

When both folds have at most 10,000 rows, permutation importance is available
with `explain_level >= 1`. Exactly 10,000 rows remains eligible. kNN SHAP is
unavailable at every dataset size because no kNN explainer is implemented.

A warning reports the training and validation row counts and the
10,000-row limit when interpretation is skipped. With `explain_level=0`, no
interpretation is requested and no size warning is emitted.

Training, predictions, evaluation metrics, and learning curves continue as usual.
Other algorithms' interpretation behavior is unchanged. The threshold is an
internal constant, not a public AutoML parameter.

## SHAP plots

The SHAP explanations are computed if `explain_level = 2`. To compute SHAP explanations the [`shap` package](https://github.com/slundberg/shap) is used.

The SHAP explanations are not available for `Baseline`, `Neural Network`, `CatBoost`, or `Nearest Neighbors`.

### SHAP importance

The SHAP importance example:

![SHAP importance](https://raw.githubusercontent.com/mljar/mljar-examples/master/Income_classification/AutoML_1/5_Default_Xgboost/learner_1_shap_summary.png)

### SHAP dependence plots

The SHAP dependence plots example:

![SHAP dependence plots](https://raw.githubusercontent.com/mljar/mljar-examples/master/Income_classification/AutoML_1/5_Default_Xgboost/learner_1_shap_dependence.png)

### SHAP decision plots

For SHAP decisions plots there are created the top-10 worst and best predictions.

The SHAP decision plots example for the best predictions:

![SHAP best decision plots](https://raw.githubusercontent.com/mljar/mljar-examples/master/Income_classification/AutoML_1/5_Default_Xgboost/learner_1_shap_class_1_best_decisions.png)

The SHAP decision plots example for the worst predictions:

![SHAP best decision plots](https://raw.githubusercontent.com/mljar/mljar-examples/master/Income_classification/AutoML_1/5_Default_Xgboost/learner_1_shap_class_1_worst_decisions.png)
