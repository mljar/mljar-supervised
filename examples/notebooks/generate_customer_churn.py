"""Refresh the churn notebook and docs downloads; --check verifies synchronization.

This standard-library maintainer tool does not train models or refresh evidence.
Edit customer_churn.py for code and the tutorial Markdown for explanations.
"""

import argparse
import json
import re
import textwrap
from pathlib import Path

from generate_concrete_strength import console_panel

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / 'examples/scripts/customer_churn.py'
PAGE = ROOT / 'docs/docs/tutorials/customer-churn.md'
ASSETS = ROOT / 'docs/docs/assets/customer-churn'


def section(page, heading):
    start = page.index('## ' + heading + '\n')
    end = page.find('\n## ', start + 3)
    return page[start:end if end != -1 else None].strip()


def notebook(source, page):
    helpers = source[source.index('import argparse'):source.index('def run(')]
    body = textwrap.dedent(source.split('def run(output, data_path, budget):\n', 1)[1]
                          .split('\n\nif __name__ == "__main__":', 1)[0])
    blocks = re.split(r'(?m)(?=^# Step \d+\.)', body)
    blocks = [b.strip() for b in blocks if b.strip()]
    if len(blocks) != 6:
        raise ValueError('Expected six teaching steps in customer_churn.py')
    cells = []

    def add(kind, value):
        cell = {'cell_type': kind, 'id': f'churn-{len(cells):02d}',
                'metadata': {}, 'source': value.strip() + '\n'}
        if kind == 'code':
            compile(value, cell['id'], 'exec')
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    intro = page.split('# Predict which customers may leave with Python', 1)[1].split('## Start in Python', 1)[0]
    intro = '# Predict which customers may leave with Python' + intro
    intro = intro.replace('../assets/customer-churn/dataset-review.md',
                          'https://supervised.mljar.com/assets/customer-churn/dataset-review/')
    add('markdown', intro)
    add('markdown', '''## Set up your notebook

Use Python 3.11 or 3.12. Install into the notebook's environment before launching it:

```bash
python -m pip install 'mljar-supervised==1.4.0b1' jupyterlab ipywidgets
python -m jupyterlab
```

Run cells from top to bottom. This notebook includes all code; it does not need a
helper script. Standard CPU algorithms use a 120-second training budget and two
threads; total time also includes imports, reports, and reload. There is no GPU
requirement. Package installation is larger than the 132 kB dataset download.
Use a new output directory each time you train. If a cell fails, fix the cause,
restart the kernel, choose a new directory, and run from the top.
''')
    add('markdown', '## 1. Inspect customers and choose inputs\n\n' + section(page, 'Choose information available before churn'))
    add('code', helpers + '\nfrom IPython.display import Image, display\n')
    add('code', 'output = Path("AutoML_CustomerChurn_Notebook").resolve()\n'
        'data_path = Path("AutoML_ChurnData/customer-churn.csv").resolve()\n'
        'budget = 120\nthread_limits = threadpool_limits(limits=2)\n' + blocks[0] + '\ndisplay(X.head())')
    add('markdown', '## 2. Reserve test profiles\n\n' + section(page, 'Keep repeated profiles together'))
    add('code', blocks[1])
    add('markdown', '## 3. Train and compare\n\n' + section(page, 'Train and compare').split('### What you will see')[0])
    add('code', blocks[2])
    add('markdown', '## 4. Evaluate probabilities and errors\n\n' + section(page, 'Understand the predictions'))
    add('code', blocks[3] + '\ndisplay(Image(filename=str(output / "confusion-matrix.png")))')
    add('markdown', '''The matrices compare the baseline and model on the same untouched rows.
Rows are actual outcomes; columns are predictions. Baseline precision is reported
as zero by convention because it flags no customers at 0.5. It is not an estimate
of the reliability of any flagged group. All scores are row-weighted.

The printed metrics are from **your run**. Compare them with the recorded tutorial
without treating that single split as a promise of future performance. Do not tune
models or thresholds against these test results.''')
    add('markdown', '## 5. Predict and reload\n\nKeep the full `automl` directory. These example inputs are held-out rows without their labels. The next cell verifies all test probabilities after loading in a separate interpreter.')
    add('code', blocks[4])
    add('markdown', '## 6. Read the report\n\nThe next cell saves versions and evaluation metrics and displays the report. Click a model name to inspect its validation metrics and settings. Validation scores use different rows from the test scores above. Importance describes model reliance, not causation.')
    add('code', blocks[5] + '\nthread_limits.restore_original_limits()\ndisplay(report)')
    add('markdown', section(page, 'Predict another CSV and reload').replace('AutoML_CustomerChurn/', 'AutoML_CustomerChurn_Notebook/'))
    adapt = section(page, 'Adapt it to your own customers').split('Next, explore')[0]
    add('markdown', adapt)
    add('markdown', '## Attribution\n\nIranian Churn [Dataset] (2020), UCI Machine Learning Repository. [DOI: 10.24432/C5JW3Z](https://doi.org/10.24432/C5JW3Z), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Snapshot retrieved 27 October 2026. The source download is unchanged; column spacing and tariff labels are normalized in memory. Full [tutorial and evidence](https://supervised.mljar.com/tutorials/customer-churn/).')
    return {'cells': cells, 'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python'}}, 'nbformat': 4, 'nbformat_minor': 5}


def measured_results():
    evidence = json.loads((ASSETS / 'evidence.json').read_text())
    m = evidence['runs']['python312']['metrics']
    a, b = m['scores']['automl'], m['scores']['baseline']
    rows = [
        ('Log loss ↓', 'log_loss'), ('ROC AUC ↑', 'roc_auc'),
        ('Average precision ↑', 'average_precision'), ('Brier score ↓', 'brier_score'),
        ('Accuracy at 0.5 ↑', 'accuracy'), ('Precision at 0.5 ↑', 'precision'),
        ('Recall at 0.5 ↑', 'recall'),
    ]
    text = 'Recorded on 27 October 2026 using **published `mljar-supervised==1.4.0b1`**,\n'
    text += 'outside the source import path. The table and images use the Python 3.12 run.\n\n'
    text += '| Held-out metric | Baseline | AutoML |\n| --- | ---: | ---: |\n'
    for title, key in rows:
        text += f'| {title} | {b[key]:.4f} | {a[key]:.4f} |\n'
    tn, fp = a['confusion_matrix'][0]
    fn, tp = a['confusion_matrix'][1]
    text += f'\nOf {tp+fn} customers who churned, the model flagged **{tp}** and missed **{fn}**.\n'
    text += f'It also flagged **{fp}** customers who stayed. Baseline precision is shown as\n'
    text += 'zero by convention: the baseline flags nobody at this threshold.\n\n'
    text += '| Python | Time in fit() | Total script time | Sampled peak combined RSS |\n| --- | ---: | ---: | ---: |\n'
    for item in evidence['runs'].values():
        metrics, res = item['metrics'], item['resources']
        text += f'| {metrics["python"]} | {metrics["fit_wall_seconds"]:.2f} s | {res["total_process_wall_seconds"]:.2f} s | {res["sampled_peak_parent_plus_children_rss_bytes"]/1024**3:.2f} GiB |\n'
    text += '\nBoth runs reproduced probabilities exactly after reload. See the evidence record\nfor each run’s scores and package versions. Hardware: Intel Core i9-10980XE, Linux,\napproximately 251 GiB host RAM. These are server measurements, not laptop or 8 GB\nguarantees. Total time includes startup and reload, using a cached dataset;\ninstallation and download are excluded. Combined process RSS was sampled every\n0.1 seconds, can count shared pages more than once, and can miss short peaks.\n'
    return text


def replace_block(page, name, text):
    start, end = f'<!-- churn-{name}:start -->', f'<!-- churn-{name}:end -->'
    before, rest = page.split(start, 1)
    _, after = rest.split(end, 1)
    return before + start + '\n' + text.strip() + '\n' + end + after


def main(check):
    source, page = SCRIPT.read_text(), PAGE.read_text()
    page = replace_block(page, 'console', console_panel(ASSETS, 'customer-churn'))
    page = replace_block(page, 'results', measured_results())
    rendered = json.dumps(notebook(source, page), indent=1, ensure_ascii=False) + '\n'
    files = {
        PAGE: page, ROOT / 'examples/notebooks/customer_churn.ipynb': rendered,
        ASSETS / 'customer_churn.py': source, ASSETS / 'customer_churn.ipynb': rendered,
        ASSETS / 'dataset-review.md': (ROOT / 'examples/datasets/customer-churn.md').read_text(),
    }
    stale = []
    for path, content in files.items():
        if not path.exists() or path.read_text() != content:
            stale.append(str(path.relative_to(ROOT)))
            if not check:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content)
    if check and stale:
        raise SystemExit('Stale churn artifacts: ' + ', '.join(stale))
    print('Churn notebook, downloads, and recorded panels are current.' if check else 'Generated churn notebook and downloads.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    main(parser.parse_args().check)
