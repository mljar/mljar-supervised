# TabM mini sizing checks

Install the optional backend from the repository root with `pip install '.[tabm]'`.
The benchmark uses synthetic scikit-learn data, isolated subprocesses, RSS
monitoring, and PyTorch CUDA allocation/reservation measurements.

```bash
python benchmarks/tabm/synthetic.py --profile smoke --device cpu --epochs 4 --output performance-results/tabm/smoke-cpu
python benchmarks/tabm/synthetic.py --profile smoke --device cuda:0 --epochs 4 --output performance-results/tabm/smoke-gpu
python benchmarks/tabm/synthetic.py --profile scaling --device cuda:0 --epochs 4 --seconds 120 --output performance-results/tabm/scaling-gpu
python benchmarks/tabm/synthetic.py --profile large --device cuda:0 --epochs 2 --seconds 300 --output performance-results/tabm/large-gpu
```

Smoke covers all four depth/width combinations (2 or 3 blocks, width 64 or 128)
for binary classification, multiclass classification and regression, plus
Friedman regression, imbalanced labels and categorical/missing inputs. All use
16 internal ensemble members and the mini architecture.

Scaling covers 10k x 50, 100k x 100, 10k x 1000, and categorical data. The large
profile uses one million rows x 50 features. All datasets split 60/20/20 into
training/validation/test: a million-row case trains on 600k rows, validates on
200k and predicts on 200k. These are resource checks, not converged accuracy
benchmarks or comparisons against RealMLP/AutoGluon.

The worker records failures (including admission rejection) in JSON. The profile
runner writes `summary.json`; inspect statuses, not only the process exit code.
The outer timeout is the fit allowance plus 120 seconds for generation,
prediction and serialization. The learner allowance is cooperative at batch
boundaries. Peak PyTorch GPU memory excludes CUDA driver/context overhead.

## Initial measurements (2026-10-12)

On Linux with an RTX 3070 (8 GiB), pytabkit 1.7.3, PyTorch 2.11.0+cu128 and
four CPU threads per worker:

- All 15 GPU smoke cases completed at four epochs (before the final validation
  metric conversion fix); final learner tests independently check validation
  log loss against scikit-learn.
- The final million-row GPU check, using two blocks of 128 units and `k=16`,
  completed two epochs in 32.3 seconds; prediction on 200k test
  rows took 9.8 seconds. Peak process RSS was
  3.10 GiB and PyTorch reserved
  1.97 GiB on the GPU.
- The final million-row CPU check did not complete its first validation checkpoint
  within a 120-second fit allowance. It rejected the candidate as designed;
  this dataset needs a larger CPU allowance. This was a time-budget limit, not
  an out-of-memory failure.

The GPU admission estimate includes additional headroom above allocator memory.
CPU/GPU profiles overlapped other checks; timings are not a controlled speed
comparison. The local result files are under `performance-results/tabm/large/`
(`gpu-final.json` and `cpu-final.json`).
