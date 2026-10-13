# Kumo and TabICLv2

`Kumo` and `TabICLv2` are separate optional algorithms in **Compete** mode.
Both use NVIDIA's [structured-data-models (SDM)](https://github.com/NVIDIA/structured-data-models)
backend and support binary classification, multiclass classification, and regression.
Kumo supports **small, medium, and large tabular** models, with small as the default.
The relational model is not included.

## Installation

Requires Python **3.11+**. Install CPU or CUDA PyTorch as described in
[Installation](../install.md#kumo-and-tabiclv2-on-cpu-or-nvidia-gpu), then install the release candidate and pinned backend:

```bash
python -m pip install "mljar-supervised[tfm]==1.4.0rc1"
python -m pip install "git+https://github.com/NVIDIA/structured-data-models.git@b97cb914e2bdcf0d165fc2c83b5fdf795aad6a69"
```

`[kumo]` and `[tabiclv2]` are aliases for the same supporting dependencies.
`[all]` includes them too. **The separate pinned SDM installation is required**:
these extras do not install SDM's placeholder PyPI release. SDM is an evolving
backend; this integration checks the tested version instead of accepting an
untested Git revision. Standard MLJAR installation does not install these models.
These integrations are included in `1.4.0rc1`; the stable `1.3.2` package does
not provide them.

## Usage

```python
from supervised import AutoML

model = AutoML(
    mode="Compete",
    algorithms=["Kumo", "TabICLv2"],
    custom_params={
        "kumo": {"device": "auto", "size": "medium", "max_context_size": 4096},
        "tabiclv2": {"device": "auto", "max_context_size": 4096},
    },
)
model.fit(X_train, y_train)
```

When the tested backend and CUDA are available, automatic Compete selection
includes both algorithms. CPU use requires explicit algorithm selection.
`device="auto"` uses CUDA when available, otherwise CPU; explicit `cpu`, `cuda`,
and `cuda:<index>` are supported. This first version does not fine-tune weights.

## What gets tuned

`num_estimators` controls the number of transformed context views combined into
one prediction. It does not train that many independent neural networks.
`max_context_size` caps how many labeled training rows each candidate retains.
Kumo's `size` selects a pretrained architecture; TabICLv2 has a fixed architecture.

The following lists are MLJAR's search space. SDM preprocessing stays at its
backend defaults; normalization, feature/class permutations, and softmax
temperature are not separate tuning parameters in this integration.


| Parameter | Allowed search values | Default |
|---|---|---|
| `num_estimators` | `[4, 8, 16]` | `8` |
| `max_context_size` | `[1024, 4096, 8192]` | `4096` |
| `size` (Kumo only) | `["small", "medium", "large"]` | `"small"` |

Parameters supplied in `custom_params` are fixed values and are removed from
search variation. For example:

```python
custom_params = {
    "kumo": {"size": "large", "num_estimators": 8, "device": "cuda"},
    "tabiclv2": {"num_estimators": 8, "device": "cuda"},
}
```

Here context size can still vary. To use all registered search values, omit those
fixed overrides. Resource settings (`device`, prediction batch size, and memory
budgets) are controls, not accuracy-tuning parameters.

## Input and resource limits

Kumo tuning can try all three sizes. Fix `custom_params={"kumo": {"size": "large"}}`
to use only large (likewise for medium or small). Larger models require more memory;
the resource checks account for both checkpoint size and larger working buffers.
Candidates that exceed the budget fail without silently changing size.

Fixed overrides may use any positive integer context/ensemble size, subject to
memory admission. `predict_batch_size` defaults to 128. Estimators execute
sequentially and use uncached context inference to bound memory. There are no
learning-rate, depth, epoch, or ordinary early-stopping settings.

Only a reproducible sample of training-fold rows becomes labeled context; each
class is represented. Preprocessing uses that context only. Reports record the
original training rows, retained context rows, checkpoint identity, and resource
estimates. A million-row input does **not** mean a million-row context.

Numerical, categorical, missing, and unseen categorical values are supported.
Numerical missing values use context means with missing indicators; constant
columns are handled by SDM; entirely constant contexts are rejected. Sparse
inputs and sample weights are rejected.
Weighted fairness/boost-on-errors candidates skip these algorithms with a warning.
Kumo rejects more than 500 effective columns, including generated indicators and
category-count features, rather than silently truncating them.

Memory admission defaults to 60% of available RAM and 70% of free VRAM. Fixed
`memory_limit_gb`, `gpu_memory_limit_gb`, `memory_fraction`, and
`gpu_memory_fraction` overrides are available. Estimates are conservative
heuristics, not guaranteed maximum dataset sizes. Reduce context or query batch
size if a candidate exceeds its budget. Prediction time grows with query rows
and context size. Deadlines are checked between query/estimator batches;
an individual kernel or download cannot be interrupted by these checks.

Classification returns probabilities in MLJAR class order. Regression returns
the mean of the predicted quantiles, following NVIDIA's benchmark adapter.
Sampled permutation importance is supported; SHAP is disabled.

## Downloads, licenses, and portable models

Installing packages does not download model weights. First fit downloads only
the required size/task checkpoint from Hugging Face, pinned to these revisions:

| Model | Repository | Revision | Weight license |
|---|---|---|---|
| Kumo | [nvidia/Kumo-Tabular](https://huggingface.co/nvidia/Kumo-Tabular) | `3c3e10bbdb590ace29e7026847f92db3c603096d` | OpenMDW-1.1 |
| TabICLv2 | [jingang/TabICL](https://huggingface.co/jingang/TabICL) | `4dcd344ece2c00be9e831fdd35bed57b5ad83e19` | BSD-3-Clause |

Both licenses permit commercial use subject to their terms. SDM code has its own
Apache/BSD notices. Shared weights, model cards, and license notices are copied
into `sdm_weights` in the results directory, with separate Kumo size subdirectories.
Retain this directory when moving or sharing results. Redistribution of these weights requires retaining the notices.

Saved learners contain **selected training rows and labels** as context. Your
data is processed locally and is not uploaded to Hugging Face. Reload uses the
saved weights without network access. Set `local_files_only=True` to require
cached checkpoints for initial fitting; `cache_dir` selects the HF cache.
Pre-download by running a small fit for each task/model with internet access.

## Synthetic examples

The [runnable sklearn example](https://github.com/mljar/mljar-supervised/blob/master/examples/scripts/sdm_synthetic.py)
uses `make_classification` for binary/multiclass tasks and `make_friedman1` for
regression. It adds categorical values and missing data, holds out a test set,
and verifies saved-model reload. Run it from a repository checkout:

```bash
python examples/scripts/sdm_synthetic.py --device cuda --rows 1000
# CPU:
python examples/scripts/sdm_synthetic.py --device cpu --rows 300
```

The example compares both algorithms on held-out synthetic data, verifies reload,
and writes a JSON summary. These checks validate integration, not TabArena rank.
