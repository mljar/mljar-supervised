---
description: Generate Mercury apps from trained MLJAR AutoML models, run them locally, publish them quickly with platform.mljar.com, or host them on your own server.
social:
  cards_layout: default/variant
---

# Apps

`mljar-supervised` can generate Mercury apps from trained `AutoML` models. You can:

- generate the app workspace with `AutoML.app()`
- start it locally with `AutoML.local_app()`
- publish it quickly with `AutoML.publish_app()`
- run or host the generated app manually on your own infrastructure

## Generate app files

The base package can export an app without Mercury installed. To run apps from
beta 1, use Python 3.11+ and install the optional runtime:

```bash
python -m pip install "mljar-supervised[apps]==1.4.0b1"
```

This adds `mercury>=3.2.12,<4`; Mercury 3.2.12 is the tested version. `[all]`
continues to mean all optional model backends. Use `mljar-supervised[all,apps]==1.4.0b1` when you need both.

Use `app()` to generate the app workspace:

```python
from supervised import AutoML

automl = AutoML(results_path="AutoML")
automl.fit(X, y)
automl.app()
```

The generated app directory contains runtime files such as:

- `predict_single.ipynb`
- `predict_batch.ipynb`
- `app_support.py`
- `mljar_app.json`
- `config.toml`
- `requirements.txt`
- `runtime.txt`
- `automl.zip`

`requirements.txt` includes only the optional model backends needed by the
selected prediction path, including ensemble and stacking inputs. It pins the
exporter's MLJAR version, the saved models' joblib version, and the exporting
environment's scikit-learn version. `mljar_app.json` records this environment.

The archive includes shared discovery transformations and selected TFM weights
and licenses, so predictions do not need the original training directory or
checkpoint downloads. Discovery's temporary training encodings are omitted.

For an unreleased checkout, install the **same locally built wheel** on the app
machine before installing the generated requirements. A public wheel with the
same version label can contain older code. For released builds, install the
generated requirements directly. Choose the appropriate CPU/CUDA PyTorch build
first if the app needs Torch; see [installation](../install.md).

Export does not fix native-library incompatibilities. Apps use the same runtime
checks as Python predictions and refuse incompatible saved models without
substituting a different model.

The default generated app title is `MLJAR AutoML`. You can provide your own title with:

```python
automl.app(title="Churn Prediction")
```

## Start locally

Use `local_app()` to generate the app, start Mercury, and open the browser:

```python
automl.local_app()
```

The local app runs in the foreground.

!!! note
    Press `Ctrl+C` to stop the local app.

## Start manually

You can also start the generated app manually. This is useful if you want to control the environment yourself or debug the generated workspace.

Generate the app:

```python
automl.app()
```

Then run Mercury in the generated app directory:

```bash
cd AutoML/app
mercury --working-dir=.
```

If Mercury is not installed in the current environment, install the generated app dependencies first:

```bash
pip install -r requirements.txt
```

## Publish quickly

Use `publish_app()` for the fastest way to put the app online:

```python
automl.publish_app()
```

This helper:

- signs in through `platform.mljar.com`
- creates a new app URL on the first publish
- reuses the last successfully published app URL on later publishes
- uploads the generated runtime files
- prints progress and friendly error messages
- remembers the last successfully published app URL

If you want to update a specific published app URL:

```python
automl.publish_app(url="https://your-app.ismvp.org")
```

The remembered URL is stored in `publish_app_state.json` inside the AutoML `results_path`.

## Publish or host manually

Using `platform.mljar.com` is simply the fastest path, but it is not required.

You can generate the app workspace with `app()` and then host it yourself:

1. Generate the app files
2. Install the generated dependencies
3. Run Mercury on your own server
4. Deploy with your own preferred hosting setup

This is a good option when you want full control over infrastructure, networking, or deployment workflow.

## Generated app behavior

The generated app can include:

- a single prediction dashboard
- batch CSV scoring
- prediction download for batch mode
- feature importance plots when available
- feature context plots for single prediction mode

## Limits

If the trained model has more than `15` features, the generated app will include batch CSV scoring only.

!!! warning
    For datasets with more than `15` features, `mljar-supervised` skips the single prediction widget dashboard and generates a batch-only app.

## Recommended usage

- use `app()` when you want the generated files
- use `local_app()` when you want the quickest local preview
- use `publish_app()` when you want the fastest hosted deployment
- use manual startup or your own server when you need full control
- regenerate the app after retraining the model

## Tutorials

If you want a beginner-friendly walkthrough after training a model, check:

- [Share a classification model as an app](../tutorials/app-classification.md)
- [Share a regression model as an app](../tutorials/app-regression.md)
