# Concrete strength dataset record

Reviewed on 26 October 2026 for the MLJAR AutoML 1.4.0 beginner example.

## Source and reuse

- Creator: I-Cheng Yeh. Original research: *Modeling of strength of
  high-performance concrete using artificial neural networks* (1998).
- Publisher: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength).
- Attribution: Yeh, I. (1998). *Concrete Compressive Strength* [Dataset].
  UCI Machine Learning Repository. <https://doi.org/10.24432/C5PK67>.
- License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), as stated
  on UCI's dataset page. Sharing and adaptation are permitted with attribution,
  a license link, and an indication of changes.
- Download: [UCI CSV](https://archive.ics.uci.edu/static/public/165/data.csv),
  41,472 bytes. No account, credentials, Excel reader, or additional downloader
  package is required. UCI also distributes the original Excel file.
- Retrieved: 26 October 2026. UCI metadata reports an update on 11 February 2024;
  the checksum below identifies the actual snapshot used here.
- CSV SHA-256: `8d4b15b6fc68cd932d745cbd663d5ceae66dd54422e99c1e4865f2936ab7e2af`.

The script downloads the CSV unchanged and verifies its checksum on each run.
Raw data is cached locally, not included in this repository. Attribution must
accompany any separately distributed data or derived dataset. Group assignments
and prediction columns are our additions, not fields supplied by UCI.

## Question and columns

Can mixture ingredients and specimen age predict measured compressive strength
for mixtures excluded from training?

One row records ingredient quantities, age, and measured strength. There are
1,030 rows, eight numeric inputs, one numeric target, and no missing values in
the reviewed CSV. There are no categorical columns or experiment identifiers.

| Columns | Role | Units |
| --- | --- | --- |
| Cement, Blast Furnace Slag, Fly Ash, Water, Superplasticizer, Coarse Aggregate, Fine Aggregate | Inputs | kg/m³ |
| Age | Input | days |
| Concrete compressive strength | Target | MPa |

Ingredient quantities and the intended specimen age can be specified before
strength measurement. The target must never be included in prediction inputs.
The observed ages range from 1 to 365 days; strength ranges from 2.33 to 82.60 MPa.

## Related samples and validation decision

Local inspection of this snapshot found:

- 427 unique combinations of the seven ingredient quantities.
- 181 combinations appearing at multiple ages.
- 25 duplicate complete rows, and 38 repeated input rows beyond their first
  occurrence. Repeated inputs can have different measured strengths.

An ordinary row split can put the same mixture at different ages in both
training and testing. For this example's question about unseen mixtures, group
rows by exact equality of all seven ingredient quantities, excluding age and
target. Keep each group together in both the outer test split and inner
validation folds. Retain all original rows; duplicate rows stay in one group.

Use `GroupShuffleSplit(test_size=0.2, random_state=42)` once to reserve 86 mixture
groups (192 rows). The remaining 341 groups (838 rows) are training data.
The 20% setting applies to groups, so it does not produce exactly 20% of rows.
Use three-fold `GroupKFold` within training for AutoML model selection. Save
the indices and assert that mixture groups never cross a split boundary.

These groups are proxies, not verified batch or experiment IDs. Exact matching
does not catch near-identical recipes, rounding differences, or shared source
experiments. The dataset lacks identifiers needed to establish independence by
laboratory, batch, or site. Do not claim that grouping eliminates all leakage or
that this single test split establishes generalization to new laboratories.

Report row-weighted MAE in MPa on the untouched test rows. More frequently
measured mixtures contribute more to this metric. Use the training-target
**median** as the constant baseline, because it minimizes absolute error.
Neither the baseline nor AutoML uses test targets during fitting or selection.
The run report shows internal validation errors; the script's final metrics
show the separate held-out test errors.

## Runnable CPU workflow

From the repository root, in a Python 3.11 or 3.12 environment:

```bash
python -m pip install 'mljar-supervised==1.4.0b1'
python examples/scripts/concrete_strength.py --output AutoML_ConcreteStrength
```

The script uses Baseline, Linear, Decision Tree, LightGBM, and an ensemble,
with two CPU threads and a 120-second training budget. It can finish before the
budget is exhausted. Total runtime also includes imports, reports, evaluation,
and reload. No optional Torch backends or model-weight downloads are needed.

Feature discovery is disabled because custom grouped validation is currently
unsupported by discovery. Golden features, KMeans features, feature selection,
and stacking are also disabled to keep this first example focused.

Outputs include `metrics.json`, `splits.json`, `test_predictions.csv`, a
predicted-versus-measured plot, three example input rows, and the complete
`automl/` directory. Open `automl/README.md` for the leaderboard and follow its
links to individual model reports. Preserve the whole directory for prediction.
The script verifies held-out predictions after loading it in a fresh interpreter.
Choose a new output directory for a new run; existing results are not overwritten.

The three example input rows come from the held-out data, with their targets
removed. They demonstrate the prediction interface, not independently collected
measurements or recommended concrete designs.

## Interpretation and remaining work

Predictions describe patterns in the collected observations. Changing inputs in
the script does not demonstrate a causal effect or qualify a mixture for use.
This workflow does not replace physical testing or engineering assessment.

Measured beta results and environment records are retained locally under
`performance-results/concrete-strength-b1/`. The notebook and full beginner
tutorial will explain these results after script validation. Laptop performance,
an 8 GB environment, Studio execution, and a beginner walkthrough remain to be
checked; server measurements cannot establish those claims.
