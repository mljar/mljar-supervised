"""Sync custom-validation script downloads and embedded code; no training.

Run from any directory. --check verifies generated files without writing them.
"""
import argparse
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / 'docs/docs/features/custom-validation.md'
ASSETS = ROOT / 'docs/docs/assets/custom-validation'


def replace(page, key, code):
    start, end = f'<!-- validation-{key}:start -->', f'<!-- validation-{key}:end -->'
    before, rest = page.split(start, 1)
    _, after = rest.split(end, 1)
    return before + start + '\n```python\n' + textwrap.dedent(code).strip() + '\n```\n' + end + after


def main(check):
    page = PAGE.read_text()
    files = {}
    for kind in ('grouped', 'time'):
        source = (ROOT / f'examples/scripts/custom_validation_{kind}.py').read_text()
        compile(source, kind, 'exec')
        files[ASSETS / f'custom_validation_{kind}.py'] = source
        split = source.split('# docs:split:start\n', 1)[1].split('    # docs:split:end', 1)[0]
        page = replace(page, kind, split)
        if kind == 'grouped':
            fit = source.split('    # Step 4.', 1)[1].split('\n', 1)[1].split('    # Step 5.', 1)[0]
            page = replace(page, 'fit', fit)
    files[PAGE] = page
    stale = []
    for path, text in files.items():
        if not path.exists() or path.read_text() != text:
            stale.append(str(path.relative_to(ROOT)))
            if not check:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text)
    if check and stale:
        raise SystemExit('Stale validation downloads/snippets: ' + ', '.join(stale))
    print('Custom-validation downloads and code snippets are current.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    main(parser.parse_args().check)
