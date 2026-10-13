# TabM mini

TabM is an optional neural model for binary classification, multiclass
classification and single-target regression. MLJAR uses **only `tabm-mini`**,
through pytabkit. The model trains from scratch and downloads no model weights.

Install the `[tabm]` or `[all]` extra with version `1.4.0rc2`, for example
`python -m pip install "mljar-supervised[tabm]==1.4.0rc2"`.
See [installation](../install.md) for CPU and NVIDIA GPU
commands. `[realmlp]`, `[tabm]` and `[all]` currently share the same dependencies;
installing any of them makes both neural models available.

TabM is supported only in **Compete mode**. Compete automatically includes it when
pytabkit and PyTorch are installed. To train only TabM mini:

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
    algorithms=["TabM"],
    total_time_limit=120,
    custom_params={"tabm": {"device": "auto"}},
)
automl.fit(X_train, y_train)
print(automl.predict(X_test[:5]))
```

For runnable binary, multiclass and regression examples using scikit-learn data:

```bash
python examples/scripts/tabm_synthetic.py --output AutoML_tabm_check
```

This short example uses CPU, eight epochs and one validation split per task.

## Tuning

MLJAR random search and hill climbing use these discrete lists:

| Parameter | Search values | Default |
|---|---|---|
| `n_blocks` | 2, 3 | 2 |
| `d_block` | 64, 128 | 128 |
| `lr` | 0.0001, 0.0003, 0.001, 0.002, 0.003 | 0.002 |
| `dropout` | 0, 0.1, 0.2, 0.3 | 0.1 |
| `weight_decay` | 0, 0.0001, 0.0003, 0.001, 0.01 | 0.0003 |

These are MLJAR's compact settings, not the authors' tuned defaults. Mini refers
to the ensemble architecture; it does not mean that the network is necessarily
small. Our default is two blocks of 128 units and 16 ensemble members sharing
weights. The architecture is always mini; `arch_type` is not a tuning option.

`custom_params["tabm"]` fixes overrides across all candidates. Additional controls:

| Parameter | Default |
|---|---|
| `tabm_k` | 16; 32 can be explicitly selected for a separate comparison |
| `num_emb_n_bins` | 48, reduced for small training folds |
| `d_embedding` | 16 |
| `batch_size` | `auto`, from 32 to 1024 based on training-fold size |
| `predict_batch_size` | 1024, wrapper prediction chunk size |
| `n_epochs` | 256 maximum |
| `patience` | 16 |
| `device` | `auto`: CUDA when available, otherwise CPU |
| `memory_fraction` / `gpu_memory_fraction` | 0.6 / 0.7 |
| `memory_limit_gb` / `gpu_memory_limit_gb` | No additional absolute cap (GiB) |

Mixed precision and compilation are disabled. Numerical embeddings are piecewise
linear. AdamW uses gradient clipping at norm 1.0. Each ensemble member gets its
own shuffled training examples. Classification averages probabilities, not logits.

## Data, stopping and resources

- Numeric, categorical and boolean columns are supported. Training-fold means
  fill numeric missing/infinite values, with missing indicators. Categories have
  reserved missing/unknown codes. Numerical quantile transformation, embedding
  bins and regression-target normalization are fitted on the training fold only.
- Sparse inputs and sample weights are unsupported. Explicit weighted requests
  fail. Automatic Compete selection skips TabM with a warning for weighted data;
  fairness reweighting and boost-on-errors candidates also skip it.
- MLJAR owns validation folds and outer ensembles. TabM requires a validation
  fold; it does not create a hidden split. Best validation weights are restored.
  Patience 16 stops after 17 consecutive epochs without improvement, matching the
  upstream rule. Early stopping uses the same metric mapping as [RealMLP](realmlp.md#checkpoints-and-time-limits).
- MLJAR's adapter checks the time budget between training and validation batches.
  Setup or an in-progress batch can overrun the allowance. If time expires before
  any complete validation checkpoint, the candidate fails rather than saving
  unvalidated weights.
- Admission estimates include the full train/validation tensors, architecture,
  embeddings, optimizer/checkpoint copies and per-member shuffle buffers. There
  is no fixed row limit. Estimates are conservative safeguards, not guarantees.
  Smaller batches alone do not eliminate full-dataset memory costs.
- CPU and one NVIDIA GPU are supported. Memory failures do not silently shrink
  networks, subsample rows or change devices. Predictions are chunked. Model
  files store CPU tensors and reload on CPU, including models trained on GPU.

## Explanations

Permutation importance is available with `explain_level >= 1`, using up to 1,000
validation rows. SHAP is disabled. Reports include actual validation learning
curves, the best epoch, stop reason and resource estimates. No per-epoch training
curve is fabricated. Model files use Python serialization; load trusted artifacts.

Starting with `1.4.0rc2`, a numerical feature with only one
embedding interval is recorded as a diagnostic rather than repeatedly warned
about in the console. This is valid behavior: its piecewise-linear encoding
reduces to min-max scaling. The bins, embeddings and predictions are unchanged.
Each learner's `*_training.log.json` records `single_bin_feature_count`; other
backend warnings and validation errors remain visible. TabM also uses the scoped
scikit-learn compatibility bridge described in [RealMLP](realmlp.md#explanations-and-reports).

The backend adapter is derived from Apache-2.0 licensed
[pytabkit](https://github.com/dholzmueller/pytabkit); see `LICENSE-pytabkit.txt`.
Architecture guidance comes from the [TabM authors](https://github.com/yandex-research/tabm).
