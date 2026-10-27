# README examples and media

These files support the 1.4.0 README. Numerical results come from recorded runs;
capability diagrams do not imply a performance ranking.

## Quickstart

Run in a fresh working directory, outside a source checkout:

```bash
python -m venv .venv
# Linux/macOS; on Windows use .venv\Scripts\activate
source .venv/bin/activate
python -m pip install 'mljar-supervised==1.4.0b1'
python /path/to/mljar-supervised/examples/scripts/quickstart.py
```

The script uses the 150-row Iris dataset bundled with scikit-learn: four flower
measurements, three species, 120 training rows and 30 held-out test rows. The
split is stratified with seed 42. The label mapping is 0 = setosa,
1 = versicolor, 2 = virginica. These are teaching examples, not a scientific
species-identification evaluation.

Recorded on 2026-10-27 with the published PyPI beta, Python 3.11.15 and 3.12.3,
on a Linux Intel Core i9-10980XE host. Both runs used two CPU threads, a
120-second training budget, and the four algorithms in the script. Both achieved
29/30 correct test predictions. Fresh Python processes reloaded both runs and
reproduced the three saved-row predictions.
The Python 3.12 log records 12.23 seconds inside `fit()`; this is not a laptop
runtime guarantee or a hard wall-clock budget.

- [Recorded Python 3.12 output](../../docs/docs/assets/readme/quickstart-output.txt)
- [Python 3.12 package versions](../../docs/docs/assets/readme/environment.json)
- Full local models/logs: `performance-results/readme-visual-b1/py311/` and `py312/`.

The README's code blocks are copied from the marked sections of
[quickstart.py](../scripts/quickstart.py). Check them with:

```bash
python examples/readme/check_readme.py
```

The jar header added to the development checkout is not present in published
1.4.0b1. Do not insert it into the recorded beta output. Recapture the output
when the header ships in a release.

## Feature engineering and fairness

Run the standalone [example generator](run_feature_examples.py) in a fresh output
folder with the same beta installed:

```bash
python /path/to/mljar-supervised/examples/readme/run_feature_examples.py
```

It creates three saved AutoML runs, their HTML reports, and `measurements.json`.
It uses synthetic data to demonstrate API behavior, not a domain benchmark.

Feature engineering uses 500 rows with length, width, and material inputs. The
target is length × width plus a material effect and noise. The holdout contains
125 rows; the remaining 375 rows use one fixed 80/20 training/validation split.
The Linear models have matching algorithm settings; each transformed model has
its own learned coefficients. Original, golden-feature, and compact-discovery
validation RMSE values are 3.660, 1.959, and 0.473. The selected model's held-out
MAE is 0.482. The chart shows validation RMSE, not test MAE. Transformation search
uses additional compute; this is not an equal-compute comparison.

Fairness uses 800 synthetic rows, two artificial groups, and a correlated proxy.
Standard and mitigation runs use the same 600 training rows and 200 test rows.
Both use Linear and the saved AutoML decision threshold for that fitted model.
The mitigation run requests demographic parity ratio >= 0.8. On test rows:

| Measurement | Standard | With mitigation |
| --- | ---: | ---: |
| Accuracy | 0.775 | 0.740 |
| Group A positive prediction rate | 0.410 | 0.500 |
| Group B positive prediction rate | 0.860 | 0.820 |
| Demographic parity ratio | 0.477 | 0.610 |

The ratio is the smaller positive prediction rate divided by the larger one.
The test ratio remains below the chosen threshold. These artificial groups have
no demographic meaning; the experiment does not demonstrate real-world fairness.
The fairness training search is given additional mitigation opportunities, not
forced to use the same amount of compute as the standard run.

- [Measurements](../../docs/docs/assets/readme/measurements.json)
- [Training log](../../docs/docs/assets/readme/features-fairness-output.txt)
- [Feature leaderboard](../../docs/docs/assets/readme/feature-leaderboard.csv)
- [Generated feature definitions](../../docs/docs/assets/readme/feature-transformations.md)

Regenerate the charts and editable SVG diagrams from a recorded run:

```bash
python examples/readme/render_media.py performance-results/readme-visual-b1
```

Rendering needs matplotlib, numpy, and pandas. The charts have a white background,
MLJAR colors, explicit values, and text labels; color is not the only distinction.

## Media sources

| Asset | Source and interpretation |
| --- | --- |
| Customer-churn run report | Existing beta tutorial screenshot; [capture evidence](../../docs/docs/assets/customer-churn/media-evidence.json) |
| Concrete-strength feature importance | Existing beta model report; [capture evidence](../../docs/docs/assets/concrete-strength/media-evidence.json) |
| Feature comparison | Measured synthetic beta run, rendered by `render_media.py` |
| Fairness comparison | Measured synthetic beta run, rendered by `render_media.py` |
| Model families | Editable SVG capability diagram; not a benchmark or screenshot |
| App workflow | Editable SVG workflow diagram; not a screenshot |
| Tutorial preview charts | Existing measured concrete-strength and customer-churn tutorials |

The Iris app was exported with published beta 1 and opened locally with Mercury
3.2.13. Its single-row page loaded four numeric controls and displayed class 1
with probability 0.988 for the initial input. Browser screenshot capture failed,
so no app screenshot is claimed or fabricated. The app files and server log are
under `performance-results/readme-visual-b1/`.

Still to capture: a real app screenshot and short single/batch walkthrough, and
a measured foundation-model comparison on a suitable dataset. The README uses
capability diagrams until that evidence is available. Studio media also remains
part of the separate release tutorial plan.

## Validation

The custom metric snippet was checked by hand (3.5 without weights, 3.75
with weights `[3, 1]`) and through a small beta fit/predict run. Structured
report access returned a dictionary. The strict documentation build passed,
and 57 distinct README targets were checked against repository files and the
local docs build. This is not a live deployment check.

The local browser loaded all nine README images. Screenshot capture failed,
and the viewport override did not change the reported width, so visual mobile
and dark-mode review remain open. The source charts were inspected separately.

## README header

The header combines the official MLJAR vector mark from
`website-mljar/public/images/logo/logo_blue_white.svg` with an AutoML wordmark.
The copied mark is at `docs/docs/images/brand/mljar-mark.svg`; only editor metadata
was removed. Light and dark variants use the same logo geometry.

Regenerate both lockups and the leaderboard preview with:

```bash
python examples/readme/render_header.py
```

The preview reads `docs/docs/assets/readme/quickstart-leaderboard.csv`, copied
from the Python 3.12 beta quickstart run. It sorts by validation log loss; it is
an illustrated table of actual results, not a screenshot. The PyPI badge links
to the beta used in the README; the tests badge shows the live workflow status.
