"""Render the README's model-family cards as an editable SVG."""
from pathlib import Path
from xml.sax.saxutils import escape

# Original outline icons, drawn on a 32 × 32 grid.
ICONS = [
    '<path d="M16 9v7M7 23v-7h18v7"/><rect x="12" y="2" width="8" height="7" rx="2"/><rect x="3" y="23" width="8" height="7" rx="2"/><rect x="21" y="23" width="8" height="7" rx="2"/>',
    '<path d="M7 8l9 8-9 8M7 8l18 0-9 8 9 8H7M16 16V4m0 12v12"/><g fill="#eaf6fb"><circle cx="7" cy="8" r="3"/><circle cx="7" cy="24" r="3"/><circle cx="16" cy="16" r="3"/><circle cx="25" cy="8" r="3"/><circle cx="25" cy="24" r="3"/></g>',
    '<path d="M3 9l13-6 13 6-13 6L3 9zM3 16l13 6 13-6M3 23l13 6 13-6"/>',
]
CARDS = [
    ('Classic models', 'Linear, trees and boosting', 'Included with MLJAR AutoML', 'Included'),
    ('Neural models', 'RealMLP · TabM', 'Neural networks for tabular data', 'Optional install'),
    ('Foundation models', 'Kumo · TabICLv2', 'Start from pretrained models', 'Optional install'),
]


def render(output):
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="218" viewBox="0 0 1000 218" role="img">',
             '<title>Classic, neural, and foundation models</title>',
             '<desc>Classic models are included. RealMLP and TabM neural models, and Kumo and TabICLv2 foundation models, require optional installation.</desc>',
             '<g font-family="Arial, Helvetica, sans-serif">']
    for i, ((heading, models, description, badge), icon) in enumerate(zip(CARDS, ICONS)):
        x = 2 + i*336
        parts.extend([
            f'<rect x="{x}" y="2" width="324" height="212" rx="14" fill="#ffffff" stroke="#cbdfe9"/>',
            f'<rect x="{x+18}" y="20" width="44" height="44" rx="12" fill="#eaf6fb"/>',
            f'<g transform="translate({x+24},26)" fill="none" stroke="#0086b3" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">{icon}</g>',
            f'<text x="{x+76}" y="48" font-size="20" font-weight="700" fill="#004666">{escape(heading)}</text>',
            f'<text x="{x+20}" y="102" font-size="18" fill="#004666">{escape(models)}</text>',
            f'<text x="{x+20}" y="132" font-size="16" fill="#52697a">{escape(description)}</text>',
            f'<rect x="{x+20}" y="160" width="{90 if i==0 else 140}" height="29" rx="14" fill="{"#eaf5e7" if i==0 else "#eef4f8"}"/>',
            f'<text x="{x+34}" y="180" font-size="14" fill="{"#365e2c" if i==0 else "#36566a"}">{badge}</text>',
        ])
    parts.append('</g></svg>')
    Path(output).write_text('\n'.join(parts)+'\n')


if __name__ == '__main__':
    render(Path(__file__).resolve().parents[2] / 'docs/docs/assets/readme/model-families.svg')
