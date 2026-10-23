# RealMLP synthetic validation

Install the source checkout with `pip install '.[realmlp]'` and run:

```bash
python benchmarks/realmlp/synthetic.py --profile smoke --device cpu
python benchmarks/realmlp/synthetic.py --profile scaling --device cuda:0 --output performance-results/realmlp-gpu
```

`smoke` covers all nine allowed architectures on binary/multiclass classification
and regression, plus Friedman regression, imbalance and mixed categorical data.
`scaling` covers 10k×50, 100k×100, 10k×1000 and 50k-row categorical data.
`large` is an explicit 1M×50 opt-in. Defaults are eight epochs and 60 seconds
per fit: these are sizing checks, not final quality comparisons. Use 256 epochs
and a larger allowance for quality experiments. Each case uses deterministic
60/20/20 train/validation/test splits and a fresh subprocess.

JSON results contain quality, fit/prediction time, sampled peak process RSS,
PyTorch peak allocated/reserved VRAM and the resource estimates used for admission.
The parent applies a separate wall-clock timeout of the fit allowance plus 120
seconds for imports, generation and prediction. Failed/skipped configurations
remain in the summary; inspect their error messages before claiming support.

The data generators and the benchmark itself also need RAM. Do not run the large
profile on a small machine just to exercise rejection; unit tests simulate low
memory before any large allocation. GPU memory measured by PyTorch excludes
driver/context and other processes. Resource estimates remain heuristic.

## Initial measurements (2026-10-09)

Linux, 36 logical CPUs, 251 GiB host RAM, RTX 3070 8 GiB; Python 3.11,
pytabkit 1.7.3, PyTorch 2.11.0+cu128. The benchmark used four CPU threads.
All 30 GPU smoke cases and all ten scaling cases on each device succeeded.
Scaling used four epochs and a 30-second fit allowance, with architectures
1×32, 2×64 and 3×128. These are sizing results, not converged quality scores.

| Input shape | CPU fit seconds | GPU fit seconds | CPU peak RSS GiB | GPU peak RSS GiB | PyTorch GPU reserved GiB |
|---|---:|---:|---:|---:|---:|
| 10,000 × 50 | 2.7–3.0 | 3.2–3.4 | 1.02–1.05 | 1.70–1.73 | 0.06–0.08 |
| 100,000 × 100 | 9.6–12.3 | 10.2–11.4 | 1.74–1.84 | 1.95–1.96 | 0.66 |
| 10,000 × 1,000 | 6.9–7.7 | 4.6–4.9 | 1.69–1.79 | 1.94–1.95 | 0.74 |

Shapes describe the generated dataset before its 60/20/20 split. Fit timings
include wrapper/backend setup but exclude dataset generation. Peak RSS covers
the process, including generation and prediction. Other work was running on the
host, so these timings are not isolated CPU-versus-GPU performance comparisons.
The one-million-row results below extend these measurements.

The measurements exposed underestimated CUDA host overhead for small models;
the admission check now reserves 2 GiB of host overhead for CUDA fits, in
addition to its tensor/data estimates. This remains a heuristic to calibrate
on additional hardware.

## One-million-row profile (2026-10-12)

All six cases passed on the same host (RTX 3070 8 GiB, 251 GiB RAM).
Each binary-classification dataset has 1,000,000 rows and 50 numerical features,
with 600,000 training, 200,000 validation and 200,000 untouched test rows.
All models completed eight epochs within a 300-second learner allowance; four
CPU threads were assigned per fit. These are short sizing runs, not converged
quality benchmarks. CPU and GPU profiles ran concurrently, so timings include
contention and are not isolated hardware speed comparisons.

| Network | CPU fit seconds | GPU fit seconds | CPU peak RSS GiB | GPU peak RSS GiB | PyTorch GPU reserved GiB |
|---|---:|---:|---:|---:|---:|
| 1×32 | 82.0 | 80.6 | 2.64 | 2.80 | 0.88 |
| 2×64 | 101.9 | 94.1 | 2.76 | 2.80 | 0.88 |
| 3×128 | 146.6 | 113.8 | 2.73 | 2.80 | 0.88 |

PyTorch VRAM statistics exclude CUDA context/driver memory and other processes.
The admission estimator required roughly 6.4 GiB host RAM for CPU fits and
4.5 GiB host RAM plus 3.4 GiB free-budget VRAM for GPU fits. It intentionally
reserves more than the observed memory use. There is no fixed row/column cap,
but wider data, larger category vocabularies or more classes can exceed the
budget even with fewer rows. This profile establishes only the tested shape.

Raw JSON and per-case logs are under
`performance-results/realmlp/large-cpu-20261012/` and
`performance-results/realmlp/large-gpu-20261012/` (local, ignored artifacts).
