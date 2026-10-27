# Iranian telecom customer churn — dataset review

Reviewed 27 October 2026 for the beginner classification tutorial.

- Publisher: UCI Machine Learning Repository, donated 8 April 2020.
- Dataset: [Iranian Churn](https://www.archive.ics.uci.edu/dataset/563/iranian%2Bchurn%2Bdataset).
- Citation: *Iranian Churn* [Dataset]. (2020). UCI Machine Learning Repository.
  DOI: [10.24432/C5JW3Z](https://doi.org/10.24432/C5JW3Z).
- License: CC BY 4.0 according to UCI. Redistribution/adaptation requires attribution.
- Direct CSV: <https://archive.ics.uci.edu/static/public/563/data.csv>; no account required.
- Retrieved 2026-10-27; 131,728 bytes.
- SHA-256: `90d5fb6bd1630cd4de4b4d28fcf8b4cb92a8f6ab7484605b0799d47386f7dbe1`.
- The downloaded bytes are unchanged. The script normalizes repeated spaces in
  column names and maps tariff codes to names in memory.

## Meaning and timing

UCI describes one row per customer, inputs aggregated over the first nine months,
with churn measured at month twelve. This supplies a stated three-month planning
window. The file contains no timestamps or customer identifiers, so that timing
cannot be independently checked or used for a chronological split.

UCI defines 1 as churn, 0 as no churn; tariff 1 is pay-as-you-go and 2 contractual.
Subscription length is in months, usage in seconds; charge amount is an ordinal
band rather than currency. The source describes charge bands 0–9, but the CSV
contains 0–10. Preserve the observed values and document this discrepancy.

## Snapshot audit and selected inputs

The file contains 3,150 rows, 13 input fields, and one target; 495 rows are churn
(15.71%). There are no missing/nonfinite numeric values. There are 300 duplicate
full rows. Age has only five distinct values, so do not describe it as finely
measured age. See `FEATURES` in the script for the ten selected inputs.

Omit `Status`: active/non-active status may be too close to the outcome for the
intended prediction point, even though UCI describes all inputs as earlier data.
This is a conservative teaching choice, not proof of leakage. Omit `Customer
Value`: its calculation is undocumented on the dataset page. Omit `Age Group`
because `Age` already represents age. Retain `Age` for this teaching benchmark;
its use in a real retention policy needs domain review. This tutorial does not
perform a fairness assessment.

## Evaluation

Group by all ten retained input columns without the target. There are 2,826
profiles, with 324 repeated-input rows beyond the first and 22 profiles associated
with both labels. Do not discard conflicting labels or group by the outcome.
These groups prevent exact-profile overlap, not all forms of customer duplication.

Use the first of five shuffled stratified group folds (seed 42) as the untouched
test set. Use three shuffled stratified group folds within training for AutoML.
The outer split has 2,511 training rows (392 churn) and 639 test rows (103 churn).
It is approximately 80/20; grouping prevents an exact proportion. Groups do not
overlap across outer or inner splits; the script asserts this and saves indices.

A training-prior DummyClassifier supplies constant probabilities. Select AutoML
by validation log loss, evaluate held-out log loss, ROC AUC, average precision,
Brier score, and errors at a fixed 0.5 threshold. Do not tune that threshold or
models on the held-out results. All metrics are row-weighted, including repeated
profiles. No temporal generalization, probability calibration guarantee, or
benefit from contacting a customer is established by this one historical split.

## Installation and outputs

Base `mljar-supervised==1.4.0b1`, explicit CPU algorithms, 120-second training
budget, two threads. Grouped validation disables feature discovery. Save logs,
splits, predictions, chart, report, metrics, versions, and fresh-process reload
comparison. Measured results and resource records live under
`performance-results/customer-churn-b1/`; docs publish selected recorded evidence.
Laptop/8 GB measurements and the Studio route remain unvalidated.
