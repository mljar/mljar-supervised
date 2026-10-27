# Maintaining beginner tutorials

Each tutorial has its own standalone teaching script, generated notebook, and
download links. Generators use only the standard library and do not train models.

| Tutorial | Script | Notebook generator |
| --- | --- | --- |
| Concrete strength | `../scripts/concrete_strength.py` | `generate_concrete_strength.py` |
| Customer churn | `../scripts/customer_churn.py` | `generate_customer_churn.py` |

For churn, edit the script for code and `docs/docs/tutorials/customer-churn.md`
for explanations. Its generator reuses those explanations in the notebook and
fills measured-result and console panels from recorded evidence. Run:

```bash
python examples/notebooks/generate_customer_churn.py
python examples/notebooks/generate_customer_churn.py --check
```

Churn run evidence is in `performance-results/customer-churn-b1/`, with selected
public assets in `docs/docs/assets/customer-churn/`. Never replace evidence merely
by regenerating a notebook. The six-step script is usable without this tool.

## Concrete strength

`concrete_strength.ipynb` contains the full runnable workflow, generated from
`../scripts/concrete_strength.py`. Edit the script for code changes and
`generate_concrete_strength.py` for notebook explanations. The generator also
copies the script and notebook into the docs downloads directory.

Each new beginner tutorial should have its own standalone script under
`examples/scripts/`, with numbered teaching steps and a matching docs download.
Readers run that script directly; maintainers run this generator separately.
Do not add screenshot capture or docs-build dependencies to the teaching script.

From the repository root:

```bash
python examples/notebooks/generate_concrete_strength.py
python examples/notebooks/generate_concrete_strength.py --check
```

Keep the source notebook free of execution outputs. Execute a copy in a fresh
working directory with the published beta environment. Retain the executed
notebook, logs, reports, predictions, and environment record under
`performance-results/`. Never refresh performance claims from an unexecuted
notebook or substitute synthetic screenshots for measured output.

Tutorial titles should lead with the task a beginner wants to complete. This
example is titled “Predict a number from a CSV with Python”, with concrete
strength as the domain story. Start with the question and actual data rows,
then introduce terminology and the workflow. Use recorded console output and
real report screenshots to help readers compare their run with the example.

The tutorial is `docs/docs/tutorials/concrete-strength.md`. Its chart and
`assets/concrete-strength/evidence.json` come from the recorded beta script
runs, identified by checksums and package versions. Regenerating notebook code
does not regenerate or revalidate those measurements.

The report images are browser captures of the actual HTML produced by beta
`AutoML.report()`. The console image typesets an unchanged saved log excerpt;
`training-output.txt` provides its accessible text. `media-evidence.json` records
source hashes, capture settings, and image hashes. The complete saved HTML report
is included for readers to explore. Local capture source and diagnostic artifacts
are under `performance-results/concrete-strength-b1/media/`.

The tutorial now presents the console excerpt as selectable HTML with a copy
button and a compact version footer. The generator fills the marked console block
from `training-output.txt` and the Python 3.12 entry in `evidence.json`; edit
those recorded sources only when replacing them with a verified run. The
historical image remains available, but the page uses the accessible text panel.
