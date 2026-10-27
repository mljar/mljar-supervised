---
title: Get started
description: Build machine learning models from tabular data. Start with a tested Python tutorial, compare results, read reports, and reuse your saved model.
hide:
  - toc
---

<div class="mljar-hero" markdown="1">
<div markdown="1">

<p class="mljar-eyebrow">MLJAR AutoML · Python</p>

# Build machine learning models from your tabular data

Train and compare models, read their reports, and save the results for future
predictions. Start with a Python script or use a conversational notebook in
MLJAR Studio.

[Find your first example](examples.md){ .md-button .md-button--primary }
[Installation guide](install.md){ .md-button }

</div>
<div class="mljar-leaderboard">
  <h2>AutoML leaderboard</h2>
  <table class="mljar-leaderboard__table">
    <caption>Customer churn · validation results</caption>
    <thead><tr><th scope="col">Model</th><th scope="col">Log loss ↓</th></tr></thead>
    <tbody>
      <tr><th scope="row">Baseline</th><td>0.433171</td></tr>
      <tr><th scope="row">Decision Tree</th><td>0.312209</td></tr>
      <tr><th scope="row">Linear</th><td>0.232629</td></tr>
      <tr><th scope="row">LightGBM</th><td>0.134072</td></tr>
      <tr class="mljar-leaderboard__best"><th scope="row">Ensemble <span class="mljar-leaderboard__badge">Best</span></th><td>0.132638</td></tr>
    </tbody>
  </table>
</div>
</div>

## Choose your starting point

<div class="mljar-cards mljar-cards--two" markdown="1">
<div class="mljar-card" markdown="1">

### :fontawesome-brands-python: Start in Python

Follow a standalone script or notebook from data loading to evaluation and saved
predictions. The beginner examples use standard CPU models.

[Run your first tutorial](tutorials/concrete-strength.md#start-in-python)

</div>
<div class="mljar-card" markdown="1">

### :material-notebook-outline: Use MLJAR Studio

Work in a conversational notebook with visible Python code. Use the tutorial's
notebook as the starting point and ask for explanations as you go.

[Explore MLJAR Studio](https://mljar.com/studio/) · [Get a tutorial notebook](examples.md#start-here)

<p class="mljar-meta">The Python examples are tested; their Studio walkthroughs are still being validated.</p>

</div>
</div>

## Learn with a real question

<div class="mljar-cards mljar-cards--two" markdown="1">
<div class="mljar-card mljar-example" markdown="1">

[![Measured versus predicted concrete strength from the recorded tutorial run.](assets/concrete-strength/predicted-vs-measured.png)](tutorials/concrete-strength.md)

<p class="mljar-eyebrow">Start here · Engineering · Regression</p>

### :material-office-building-outline: Predict a number from a CSV

Estimate concrete strength from its ingredients and age. Compare prediction
errors in MPa, read the report, and reload your saved model.

[Follow the concrete-strength tutorial →](tutorials/concrete-strength.md)

</div>
<div class="mljar-card mljar-example" markdown="1">

[![Baseline and AutoML classification errors on held-out customer records.](assets/customer-churn/confusion-matrix.png)](tutorials/customer-churn.md)

<p class="mljar-eyebrow">Start here · Business · Classification</p>

### :material-account-arrow-right-outline: Predict which customers may leave

Estimate churn probabilities from customer usage. Learn about missed customers,
false alarms, and the difference between a probability and a decision.

[Follow the customer-churn tutorial →](tutorials/customer-churn.md)

</div>
</div>

[Browse examples and advanced guides](examples.md){ .md-button }

<span id="whats-good-in-it"></span>

## What you can do with AutoML

<div class="mljar-cards mljar-cards--three" markdown="1">
<div class="mljar-card" markdown="1">

### :material-file-chart-outline: Read automatic reports

Compare models in the leaderboard and inspect their settings, evaluation,
and available explanations.

[Read a report](tutorials/concrete-strength.md#read-the-reports)

</div>
<div class="mljar-card" markdown="1">

### :material-auto-fix: Generate features

Try golden features and feature discovery, then check whether they improve
validation results. Discovery is included in the base installation.

[Explore feature discovery](features/feature_discovery.md)

</div>
<div class="mljar-card" markdown="1">

### :material-brain: Try foundation models

Use optional Kumo and TabICLv2 tabular foundation models. Check dependencies,
first-use weight downloads, and hardware requirements.

[Explore TFMs](features/sdm.md)

</div>
<div class="mljar-card" markdown="1">

### :material-scale-balance: Evaluate fairness

Measure differences between groups and assess bias mitigation alongside
predictive performance. The choice of groups and metrics matters.

[Explore fairness and mitigation](tutorials/fairness-binary-classification.md)

</div>
<div class="mljar-card" markdown="1">

### :material-application-outline: Export a prediction app

Create a web interface for individual rows and CSV files. Export files from the
base package; add the optional runtime to run the app.

[Create an app](features/apps.md)

</div>
<div class="mljar-card" markdown="1">

### :material-tune: Customize your workflow

Choose a training mode, define an evaluation metric, or supply validation splits
that reflect your data's groups and timing.

[Find advanced guides](examples.md#advanced-guides)

</div>
</div>

## Installation

In a Python 3.11+ environment, install the exact beta used by the new tutorials:

```bash
python -m pip install 'mljar-supervised==1.4.0b1'
```

The beginner workflows were tested on Python 3.11 and 3.12. See the
[installation guide](install.md) for environment setup, supported platforms,
and optional neural models, TFMs, and app dependencies.

## Basic usage

Download [customer_churn.py](assets/customer-churn/customer_churn.py){ download="customer_churn.py" }
into your working folder, then run:

```bash
python customer_churn.py --output AutoML_CustomerChurn
```

The script downloads the reviewed data, trains models, evaluates held-out rows,
and verifies predictions after reload. Open
`AutoML_CustomerChurn/automl/README.html` to read your report.
[Follow the complete explanation](tutorials/customer-churn.md).

## Keep going

[Save and reload models](features/save-and-load-models.md) ·
[API reference](api.md) · [Get help](get_help.md) ·
[Contribute](contributing.md) · [Source and license](https://github.com/mljar/mljar-supervised)
