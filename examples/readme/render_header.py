"""Render the README brand lockups and its recorded Iris leaderboard.

Uses the official MLJAR vector logo and the saved beta leaderboard.
No models are trained and no benchmark values are invented.
"""
import copy
import csv
from pathlib import Path
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / 'docs/docs/assets/readme'
NS = 'http://www.w3.org/2000/svg'
ET.register_namespace('', NS)
ET.register_namespace('xlink', 'http://www.w3.org/1999/xlink')
mark = ET.parse(ROOT / 'docs/docs/images/brand/mljar-mark.svg').getroot()
for variant, color in [('light', '#004666'), ('dark', '#e7f5fc')]:
    svg = ET.Element(f'{{{NS}}}svg', {'width':'360', 'height':'104', 'viewBox':'0 0 360 104', 'role':'img'})
    ET.SubElement(svg, f'{{{NS}}}title').text = 'MLJAR AutoML'
    logo = copy.deepcopy(mark)
    logo.attrib.update({'x':'4','y':'8','width':'88','height':'88'})
    svg.append(logo)
    ET.SubElement(svg, f'{{{NS}}}text', {'x':'112', 'y':'72', 'fill':color,
        'font-family':'Arial, Helvetica, sans-serif', 'font-size':'56', 'font-weight':'700'}).text = 'AutoML'
    ET.ElementTree(svg).write(ASSETS / f'header-{variant}.svg', encoding='unicode')

with (ASSETS/'quickstart-leaderboard.csv').open() as f:
    rows = sorted(csv.DictReader(f), key=lambda r: float(r['metric_value']))
parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="760" height="342" viewBox="0 0 760 342" role="img">',
         '<title>Iris quickstart leaderboard: Ensemble has the lowest validation log loss, 0.111935.</title>',
         '<rect x="5" y="7" width="750" height="330" rx="16" fill="#dbe8ef" opacity="0.45"/>',
         '<rect x="1" y="1" width="750" height="330" rx="16" fill="#ffffff" stroke="#d5e3ec"/>',
         '<g font-family="Arial, Helvetica, sans-serif" fill="#004666">',
         '<text x="28" y="38" font-size="21" font-weight="700">Iris · model comparison</text>',
         '<text x="28" y="63" font-size="14" fill="#52697a">One training run. A report for every model.</text>',
         '<path d="M28 79H724" stroke="#d5e3ec"/>',
         '<text x="28" y="104" font-size="14" font-weight="700">MODEL</text>',
         '<text x="724" y="104" font-size="14" font-weight="700" text-anchor="end">VALIDATION LOG LOSS ↓</text>']
for i, row in enumerate(rows):
    y = 134+i*36
    if i==0: parts.append(f'<rect x="16" y="{y-19}" width="720" height="33" rx="6" fill="#eaf5e7"/>')
    parts.append(f'<text x="28" y="{y+3}" font-size="16">{escape(row["name"])}</text>')
    parts.append(f'<text x="724" y="{y+3}" font-size="16" text-anchor="end">{float(row["metric_value"]):.6f}</text>')
parts.extend(['<text x="28" y="312" font-size="13" fill="#52697a">Recorded with MLJAR 1.4.0b1 · lower is better</text>', '</g></svg>'])
(ASSETS/'quickstart-leaderboard.svg').write_text('\n'.join(parts)+'\n')

# Image links keep the call-to-action styling on GitHub without custom CSS.
buttons = [
    ('quickstart', 'Quickstart', 144, True,
     '<path d="m2 4 5 5-5 5M10 14h6"/>'),
    ('tutorials', 'Tutorials', 144, False,
     '<path d="M9 4v13M9 5C6 2 3 2 1 3v12c3-1 5 0 8 2 3-2 5-3 8-2V3c-2-1-5-1-8 2z"/>'),
    ('studio', 'MLJAR Studio', 172, False,
     '<path d="M3 2h12a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H7l-5 4V4a2 2 0 0 1 1-2zM6 6h7M6 10h5"/>'),
]
for slug, label, width, primary, icon in buttons:
    fill = '#007fa6' if primary else '#ffffff'
    ink = '#ffffff' if primary else '#004666'
    border = fill if primary else '#bfd7ea'
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="42" viewBox="0 0 {width} 42" role="img">
<title>{escape(label)}</title>
<rect x="1" y="1" width="{width-2}" height="40" rx="8" fill="{fill}" stroke="{border}"/>
<g transform="translate(16,12)" fill="none" stroke="{ink}" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">{icon}</g>
<text x="45" y="26" font-family="Arial, Helvetica, sans-serif" font-size="14" font-weight="700" fill="{ink}">{escape(label)}</text>
</svg>'''
    (ASSETS/f'button-{slug}.svg').write_text(svg+'\n')
