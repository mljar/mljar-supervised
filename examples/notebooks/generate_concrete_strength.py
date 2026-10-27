"""Generate the teaching notebook and docs downloads from the tested script.

Run from any directory; --check verifies that generated files are current.
Only the Python standard library is required. Committed notebooks omit outputs;
executed notebooks and their complete artifacts belong in performance-results/.
"""

import argparse
import html
import json
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "examples/scripts/concrete_strength.py"
NOTEBOOK = ROOT / "examples/notebooks/concrete_strength.ipynb"
DOWNLOADS = ROOT / "docs/docs/assets/concrete-strength"
TUTORIAL = ROOT / "docs/docs/tutorials/concrete-strength.md"


def console_panel(downloads=DOWNLOADS, asset_slug="concrete-strength"):
    """Keep the displayed log and footer tied to recorded evidence."""
    evidence = json.loads((downloads / "evidence.json").read_text())
    run = evidence["runs"]["python312"]
    metrics = run["metrics"]
    lines = (downloads / "training-output.txt").read_text().splitlines()
    styled = []
    for line in lines:
        escaped = html.escape(line)
        if line.startswith("* Step"):
            escaped = f'<span class="training-console__step">{escaped}</span>'
        elif line.startswith("AutoML best model:"):
            escaped = f'<span class="training-console__result">{escaped}</span>'
        styled.append(escaped)
    output = "\n".join(styled) + "\n"
    return (
        '<figure class="training-console" aria-label="Recorded AutoML training output">\n'
        '<figcaption class="training-console__header">'
        '<span class="training-console__title"><span aria-hidden="true">&gt;_</span> Training output</span>'
        '<button type="button" class="training-console__copy" data-console-copy hidden '
        'aria-label="Copy training output">'
        '<svg class="training-console__icon" width="16" height="16" fill="none" '
        'stroke="currentColor" stroke-width="1.75" viewBox="0 0 24 24" aria-hidden="true" focusable="false">'
        '<rect x="8" y="8" width="12" height="12" rx="2"/>'
        '<path d="M16 8V5a1 1 0 0 0-1-1H5a1 1 0 0 0-1 1v10a1 1 0 0 0 1 1h3"/>'
        '</svg><span data-console-copy-label>Copy</span></button></figcaption>\n'
        '<pre tabindex="0" aria-label="Training log"><code>' + output + '</code></pre>\n'
        '<footer class="training-console__footer"><span>'
        f'MLJAR {html.escape(metrics["package_versions"]["mljar-supervised"])} · '
        f'Python {html.escape(metrics["python"])}</span>'
        f'<a class="training-console__report" href="../../assets/{asset_slug}/report.html">'
        '<svg class="training-console__icon" width="16" height="16" fill="none" '
        'stroke="currentColor" stroke-width="1.75" viewBox="0 0 24 24" aria-hidden="true" focusable="false">'
        '<path d="M14 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"/>'
        '<path d="M14 3v6h6M8 17v-3m4 3v-5m4 5v-2"/>'
        '</svg><span>Full report</span></a>'
        '<span class="training-console__status" role="status" aria-live="polite"></span>'
        '</footer></figure>'
    )


def notebook(source):
    imports = source[source.index("import argparse"):source.index("def run(")]
    body = textwrap.dedent(source.split("def run(output, data_path, budget):\n", 1)[1]
                           .split('\n\nif __name__ == "__main__":', 1)[0])
    boundaries = [
        "# Keep the same mixture", "# A training median",
        "# Step 4.", "# A new interpreter",
        "# Step 6.", "metrics = {", "# Step 7.",
    ]
    positions = [0] + [body.index(marker) for marker in boundaries] + [len(body)]
    blocks = [body[a:b].strip() for a, b in zip(positions, positions[1:])]
    cells = []

    def cell(kind, value):
        if kind == "markdown":
            value = textwrap.dedent(value)
        entry = {"cell_type": kind, "id": f"concrete-{len(cells):02d}",
                 "metadata": {}, "source": value.strip() + "\n"}
        if kind == "code":
            entry.update(execution_count=None, outputs=[])
        cells.append(entry)

    cell("markdown", """
    # Predict a number from a CSV with Python

    **Your first AutoML regression model: concrete strength**

    Imagine you have a spreadsheet of concrete test results. Each row describes
    the ingredients, the specimen's age, and its measured strength. For another
    mixture, you know the ingredients and age but want an estimate of strength.
    Can a model learn from those past measurements to make that prediction?

    Predicting a number is called **regression**. In this notebook, you will look
    at the data, train several models with AutoML, and compare their predictions
    with measurements that were kept out of training. Then you will read the
    reports and reload the saved model. Basic familiarity with running Python
    is enough to begin; we explain the machine-learning concepts as we use them.

    ## A first look at the data

    These are the first five rows of the real CSV, showing four of its nine columns:

    | Cement (kg/m³) | Water (kg/m³) | Age (days) | Measured strength (MPa) |
    | ---: | ---: | ---: | ---: |
    | 540.0 | 162.0 | 28 | 79.99 |
    | 540.0 | 162.0 | 28 | 61.89 |
    | 332.5 | 228.0 | 270 | 40.27 |
    | 332.5 | 228.0 | 365 | 41.05 |
    | 198.6 | 192.0 | 360 | 44.30 |

    The first row describes a mixture with 540 kg of cement and 162 kg of water
    per cubic metre, tested at 28 days, with measured strength 79.99 MPa.
    The other five ingredient columns are omitted from this short preview;
    for example, the first two rows have different coarse aggregate amounts.
    The full table appears after loading below.

    We give the model eight **inputs**, also called *features*: seven ingredient
    quantities and age. We want it to predict one **target**: strength in MPa
    (megapascals). Do not include that measured target among the prediction inputs.

    ## Set up your notebook

    Use Python 3.11 or 3.12. Before starting this notebook, install into its Python
    environment: `python -m pip install 'mljar-supervised==1.4.0b1' jupyterlab ipywidgets`.
    Start Jupyter with `python -m jupyterlab`. No GPU or optional Torch packages
    are needed. The UCI CSV download is 41.5 kB; package installation is larger.

    This notebook uses a 120-second training budget and two CPU threads. Training
    can finish early; total runtime also includes imports, reports, and reload.
    Script runs on the recorded Linux server took 16–21 seconds. Laptop and 8 GB
    measurements are still pending. Your environment can produce different timings.

    Run cells in order. Use a new output directory for each run. The code below
    is generated from `examples/scripts/concrete_strength.py`; it is fully
    included here, so you do not need that file to execute this notebook.
    """)
    cell("markdown", """
    ## 1. Load and inspect the measurements

    One row contains seven ingredient amounts (kg/m³), specimen age (days), and
    measured compressive strength (MPa). The **target** is the quantity to predict:
    `Concrete compressive strength`. All other columns are numeric inputs.

    Source: Yeh, I. (1998), *Concrete Compressive Strength*,
    [UCI Machine Learning Repository](https://doi.org/10.24432/C5PK67),
    [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
    We download UCI's CSV unchanged and check the reviewed file's checksum.
    The 1,030 rows contain no missing values. Group labels and predictions are
    additions made by this example. A changed checksum stops the example so a
    changed source cannot silently alter the results.
    """)
    cell("code", imports + "\nfrom IPython.display import display\n")
    cell("code", 'output = Path("AutoML_ConcreteStrength_Notebook").resolve()\n'
         'data_path = Path("AutoML_ConcreteData/concrete.csv").resolve()\n'
         'budget = 120\nthread_limits = threadpool_limits(limits=2)\n' + blocks[0]
         + '\ndisplay(data.head())')
    cell("markdown", """
    ## 2. Reserve mixtures for testing

    Some mixtures were measured at multiple ages. A random row split could put
    the same recipe in training and testing. Here we ask about **unseen mixtures**,
    so we group rows by the seven ingredient amounts and keep each group together.

    We reserve 86 groups (192 rows) for final testing. The remaining 341 groups
    (838 rows) train the models. Within training, three grouped folds rotate which
    rows train and validate each candidate. AutoML uses those validation results
    to select a model. The held-out test target is used only for final evaluation.

    These exact ingredient matches are proxy groups, not verified experiment IDs.
    Near-identical recipes or shared laboratories may still cross the split.
    No batch/site identifiers are available to check those relationships.
    """)
    cell("code", blocks[1])
    cell("markdown", """
    ## 3. Train a baseline and compare models

    A baseline always predicts the median training strength. The median minimizes
    constant absolute error, giving us a useful reference for **MAE**, the mean
    absolute difference between a prediction and its measured value.

    AutoML compares Baseline, Linear, Decision Tree, and LightGBM using the three
    training folds. It also tries an ensemble that combines candidate predictions.
    `n_jobs=2` limits model parallelism, and `total_time_limit=120` assigns a
    training budget. Keep the seed and split fixed when comparing configurations.

    Feature discovery is included in the base package, but currently does not
    support custom grouped validation, so it is disabled here. Other feature
    engineering and stacking are also disabled to keep this first run focused.
    The `Baseline` in AutoML's leaderboard uses its own implementation; the
    separately fitted median is our held-out evaluation reference.
    """)
    cell("code", blocks[2])
    cell("markdown", """
    ## 4. Evaluate on the held-out rows and predict example inputs

    These rows were not used for training or model selection. We save predictions
    with their measurements for inspection. The three example inputs are held-out
    rows with the target removed, not proposed concrete designs or new experiments.
    """)
    cell("code", blocks[3])
    cell("markdown", """
    The recorded beta script runs produced **4.30 MPa MAE**, compared with
    **12.59 MPa** for the training-median baseline. A 4.30 MPa MAE means an average
    absolute error of 4.30 MPa across these test rows; it does not bound every error.
    Whether this is useful depends on the intended use and acceptable errors.
    Mixtures with more measured rows contribute more to this row-weighted metric.

    Do not repeatedly tune against this test result. Use training validation for
    changes and reserve fresh evaluation data for a new final assessment.
    """)
    cell("markdown", """
    ## 5. Reload in a fresh Python process

    Keep the entire `automl/` directory, including preprocessing and all ensemble
    dependencies. This check starts a separate interpreter, reads the saved model
    and input CSV, then verifies that predictions agree with the in-memory model.
    It raises an error if they differ beyond a small numerical tolerance.
    """)
    cell("code", blocks[4])
    cell("markdown", """
    ## 6. Inspect errors and save the run record

    Points near the diagonal have predictions close to measured strength. Look
    for large errors and ranges where predictions tend to be too high or low.
    The example saves the figure, metrics, split indices, and package versions.
    These records identify the run behind any reported number or chart.
    """)
    cell("code", blocks[5] + '\nfrom IPython.display import Image, display\n'
         'display(Image(filename=str(output / "predicted-vs-measured.png")))')
    cell("code", blocks[6] + '\nthread_limits.restore_original_limits()')
    cell("markdown", """
    ## 7. Read the reports

    Run the next cell to display the interactive HTML report here. Click a model
    name to open its individual report and **Go back** to return to the overview.
    This also writes `AutoML_ConcreteStrength_Notebook/automl/README.html`, which
    you can open in a browser. The Markdown version is available as `README.md`.
    The leaderboard compares **training validation** MAE, not the separate test
    score above. Lower MAE is better. In the recorded run the selected ensemble
    had validation MAE 4.66 MPa and test MAE 4.30 MPa, measured on different rows.

    Follow the `4_Default_LightGBM` link to see its settings, validation metrics,
    learning curves, permutation importance, and residual plots. Importance
    describes model dependence on inputs; it does not establish causal effects.
    The exact artifacts available depend on the algorithm and explanation level.
    """)
    cell("code", blocks[7] + '\ndisplay(report)')
    cell("markdown", """
    ## 8. Adapt this workflow to your CSV

    1. Replace `load_data(data_path)` with `pd.read_csv("your_data.csv")`; the UCI
       checksum and schema checks apply only to the concrete dataset.
    2. Set `TARGET` to your measured outcome and `FEATURES` to inputs available
       when predicting. For this regression example, the outcome must be numeric.
    3. Replace `INGREDIENTS` and the grouping rule with your real experiment,
       customer, batch, or subject identifier. Keep identifiers out of model
       inputs unless their use is justified. Time-dependent data needs a time
       split; do not blindly reuse this mixture split.
    4. Choose an untouched test set, a baseline, a budget, and an error measure
       meaningful in your units. Set a new output path before running.
    5. Inspect errors, read reports, and verify prediction/reload with the same
       input columns and units. Domain knowledge determines acceptable errors.

    To predict a later CSV in a new session:

    ```python
    import pandas as pd
    from supervised import AutoML
    model = AutoML(results_path="AutoML_ConcreteStrength_Notebook/automl")
    new_rows = pd.read_csv("AutoML_ConcreteStrength_Notebook/new_rows.csv")
    predictions_mpa = model.predict(new_rows)
    ```

    If the output directory already exists, choose a new name to retrain or load
    the existing model as above. If a notebook cell fails, fix the cause, restart
    the kernel, choose a new output directory, and run from the top.

    **Limits:** exact recipe grouping cannot establish independence by experiment
    or laboratory. A single split does not establish general performance.
    Observational predictions do not prove that changing an ingredient causes
    a strength change or replace physical testing and engineering assessment.

    **Studio starting prompt (not yet validated in Studio):** “Help me run this
    concrete-strength notebook. Explain the target and mixture groups, compare
    the held-out error with the baseline, and show the saved model report.”
    The Python/Jupyter workflow is tested separately; Studio execution remains
    part of the release checklist.
    """)
    return {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }, "nbformat": 4, "nbformat_minor": 5}


def main(check):
    source = SCRIPT.read_text()
    rendered = json.dumps(notebook(source), indent=1, ensure_ascii=False) + "\n"
    files = {NOTEBOOK: rendered, DOWNLOADS / SCRIPT.name: source,
             DOWNLOADS / NOTEBOOK.name: rendered}
    page = TUTORIAL.read_text()
    start, end = "<!-- concrete-console:start -->", "<!-- concrete-console:end -->"
    before, rest = page.split(start, 1)
    _, after = rest.split(end, 1)
    files[TUTORIAL] = before + start + "\n" + console_panel() + "\n" + end + after
    stale = []
    for path, content in files.items():
        if check:
            if not path.exists() or path.read_text() != content:
                stale.append(str(path.relative_to(ROOT)))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
    if stale:
        raise SystemExit("Regenerate concrete example files: " + ", ".join(stale))
    print("Concrete notebook and downloads are current." if check else "Generated concrete notebook and downloads.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    main(parser.parse_args().check)
