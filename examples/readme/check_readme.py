"""Check that README quickstart blocks match the runnable source script."""
from pathlib import Path

root = Path(__file__).resolve().parents[2]
script = (root / 'examples/scripts/quickstart.py').read_text()
readme = (root / 'README.md').read_text()
for label in ['QUICKSTART', 'RELOAD']:
    source = script.split(f'# README_{label}_START\n')[1].split(f'# README_{label}_END')[0].strip()
    block = readme.split(f'<!-- README_{label}_START -->')[1].split(f'<!-- README_{label}_END -->')[0].strip()
    assert block == f'```python\n{source}\n```', f'{label}: README differs from script'
print('README quickstart and reload blocks match the runnable script.')
