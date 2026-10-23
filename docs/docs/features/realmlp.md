# RealMLP

RealMLP is an optional neural model for binary classification, multiclass
classification and single-target regression. Install
the `[realmlp]` or `[all]` extra with version `1.4.0rc5`, for example
`python -m pip install "mljar-supervised[realmlp]==1.4.0rc5"`.
Choose [CPU](../install.md#realmlp-on-cpu) or
[NVIDIA GPU](../install.md#realmlp-on-nvidia-gpu) installation and follow the
[verification steps](../install.md#verify-the-installation). It trains from scratch
without downloading weights.

## Example with generated data

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from supervised import AutoML

X, y = make_classification(
    n_samples=1000, n_features=20, n_informative=10, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
automl = AutoML(
    mode="Compete",
    algorithms=["RealMLP"],
    total_time_limit=300,
    n_jobs=4,
    custom_params={"realmlp": {"device": "auto"}},
)
automl.fit(X_train, y_train)
print(automl.predict(X_test))
```

For multiclass data, set `n_classes=3` in `make_classification`. For regression,
use `make_regression(n_samples=1000, n_features=20, noise=10, random_state=42)`
from `sklearn.datasets` and remove `stratify=y` from the split.

The runnable [synthetic example](https://github.com/mljar/mljar-supervised/blob/master/examples/scripts/realmlp_synthetic.py)
trains and scores all three tasks on held-out data. It uses a single validation
split, one small model and eight epochs per task for a quick check:

```bash
python examples/scripts/realmlp_synthetic.py --output /tmp/realmlp-example
# For the normal training budget, use a fresh output directory:
python examples/scripts/realmlp_synthetic.py --epochs 256 --output /tmp/realmlp-full
```

The short example checks that training works; eight epochs are not a quality benchmark.

## Data and validation

- Numeric, categorical and boolean columns are supported. Numeric missing values
  and infinities are mean-imputed using the training fold; columns missing during
  fitting receive missing-value indicators. All-missing numerical columns use zero.
- Categorical mappings use observed training values with reserved codes for
  missing and unknown values. Predictions must have the training columns;
  reordered DataFrame columns are accepted.
- MLJAR performs its datetime, text and IP transformations. RealMLP owns numerical
  scaling and regression-target normalization; there is no second MLJAR scaling step.
- Sparse inputs and sample weights are unsupported. Explicit weighted RealMLP
  requests fail; automatic Compete selection skips RealMLP with a warning for
  weighted data. Generated fairness reweighting and boost-on-errors candidates
  exclude RealMLP. Unweighted fitted models are not relabeled as reweighted models.
- MLJAR provides each validation fold. Internal CV, refitting and ensembles are
  disabled. No hidden validation split is created when validation is absent.

## Parameters

The MLJAR default is **two hidden layers of 64 units**. This overrides the
authors' larger Tuned Defaults (TD) architecture. Other training defaults come
from RealMLP-TD, including learning rate 0.04 for classification and 0.2 for regression.

MLJAR random search and hill climbing use these discrete lists:

| Parameter | Allowed search values |
|---|---|
| `n_hidden_layers` | 1, 2, 3 |
| `hidden_width` | 32, 64, 128 |
| `lr` | 0.02, 0.04, 0.1, 0.2 |
| `p_drop` | 0, 0.15, 0.3 |
| `wd` | 0.001, 0.01, 0.02, 0.05 |

`custom_params["realmlp"]` fixes overrides across candidates, for example:

```python
automl = AutoML(
    mode="Compete",
    algorithms=["RealMLP"],
    custom_params={"realmlp": {
        "device": "cpu",
        "n_hidden_layers": 1,
        "hidden_width": 32,
        "n_epochs": 256,
        "memory_limit_gb": 8,
    }},
)
```

RealMLP is supported only in Compete mode. `AutoML(mode="Compete")` includes
it automatically when pytabkit and PyTorch are installed. Explicit selection
in other modes raises an error. Without the optional dependencies, Compete
keeps its existing algorithm list.

`custom_params` is shared across algorithms. For example,
`custom_params={"realmlp": {"device": "auto"}, "random_forest": {"max_depth": 3}}`
sets fixed overrides for both models. Algorithm names are case-insensitive and
ignore spaces, underscores and hyphens. Existing algorithms accept their
registered tuning/default parameters; RealMLP also accepts the resource and
training controls below. Task, metric, class count and thread controls remain
owned by AutoML. Unknown parameters and duplicate algorithm aliases are rejected.
These overrides apply to default, random and hill-climbing candidates and are
saved with AutoML settings. `custom_params` is not supported in Optuna mode yet.

## Checkpoints and time limits

With validation, RealMLP restores the best validation epoch even when early
termination is disabled (the default). Training normally lasts 256 epochs.
`use_early_stopping=True` enables this rule:

`epoch > early_stopping_multiplicative_patience * best_epoch + early_stopping_additive_patience`

Patience defaults are 3 and 40. The MLJAR learner allowance is passed to pytabkit
after wrapper preprocessing. The timer is cooperative: upstream setup, validation
and saving can cause overruns. A fit without a completed finite validation
checkpoint is rejected. Without validation, no best epoch is reported.

Checkpoint metrics are cross-entropy for logloss, AUC loss for AUC, classification
error for accuracy, RMSE for RMSE/MSE/R2, and MAE for MAE. Other metrics emit a
warning and use cross-entropy or RMSE for checkpoint selection. MLJAR still ranks
models using the requested metric. Label smoothing defaults off for logloss/AUC.

## Resource admission

Before tensor training allocation, MLJAR checks the dataset and network against
pytabkit's metadata-based RAM/VRAM estimates, with additional headroom for
preprocessing and framework overhead. Both train and validation data count.
These estimates are safeguards, not guarantees; there is no universal row limit.

| Override | Default |
|---|---|
| `device` | `auto` (`cpu`, `cuda`, `cuda:0`, etc. also supported) |
| `memory_fraction` | 0.6 of available host/container RAM |
| `gpu_memory_fraction` | 0.7 of current free VRAM |
| `memory_limit_gb` | No additional absolute cap (GiB units) |
| `gpu_memory_limit_gb` | No additional absolute cap (GiB units) |
| `batch_size` | Upstream default, 256 |
| `predict_batch_size` | Wrapper chunk size, 1024 |

`device="auto"` uses CUDA when available, otherwise CPU. Set `device="cpu"` to
force CPU training. Device settings control training, not which PyTorch packages
are installed or downloaded. Apple MPS is not enabled.

`n_jobs` respects CPU affinity/container quotas. CUDA folds run sequentially in
MLJAR. Memory failures are reported per candidate; there is no automatic
subsampling, architecture modification or CPU retry. Other eligible candidates
can continue. A smaller batch cannot fix memory needed to store the full dataset.
Predictions are chunked before upstream conversion/device transfer.

Models are saved with CPU tensors and reload on CPU, including models trained on
CUDA. Keep compatible pytabkit/PyTorch versions for loading. Model artifacts use
Python serialization and should only be loaded from trusted sources.

## Explanations and reports

`explain_level >= 1` produces validation permutation importance, using up to
1,000 sampled rows and one worker. Importance uses logloss for classification
and MSE for regression. It describes predictive importance, not causality.
SHAP is not enabled for RealMLP in this release.

Validation learning curves use the actual upstream stopping metric. Upstream
does not expose a matching per-epoch training metric here, so the train curve
is omitted. Each learner's `*_training.log.json` records resource estimates,
architecture, backend versions, best/completed validation epochs and stopping
reason. The model report also includes the training summary.

Starting with `1.4.0rc2`, routine Lightning INFO messages
(device announcements, epoch-limit messages and logging-service tips) are disabled
during MLJAR fitting and prediction. MLJAR progress and backend warnings/errors
remain visible. Scoped compatibility bridges use the current scikit-learn finite
validation API and PyTorch leaf-spec API; they restore upstream functions and
logging settings after each call, including failures. Installed dependencies
are not modified.

Portable models intentionally reload on CPU. Prediction does not emit Lightning's
GPU-available advisory for this explicit CPU inference path; other warnings remain
visible.

Completed learners release their early-stopping checkpoint copies. GPU admission
checks collect unreachable tensors and release unused CUDA cache before measuring
free VRAM. Real memory shortages still reject the candidate.

## Validation and benchmarks

The optional test suite covers generated binary/multiclass classification,
linear and Friedman regression, mixed categories/missing values, save/load,
resource rejection and CUDA-to-CPU loading. The synthetic benchmark in
`benchmarks/realmlp/synthetic.py` measures representative sizes and architectures
separately from normal CI. Large datasets are opt-in and checked before fitting.
Synthetic results verify behavior and resource use; compare real-world quality
using the separate TabArena benchmark.

Sources: [pytabkit](https://github.com/dholzmueller/pytabkit),
[RealMLP paper](https://arxiv.org/abs/2407.04491).
