"""Render the README's app-workflow cards as an editable SVG."""
from pathlib import Path
from xml.sax.saxutils import escape

ICONS = [
    '<path d="M18 3H6a2 2 0 0 0-2 2v22a2 2 0 0 0 2 2h18a2 2 0 0 0 2-2V15M18 3v8h8M18 3l8 8M12 20h17m-5-5 5 5-5 5"/>',
    '<rect x="5" y="4" width="22" height="18" rx="2"/><path d="M2 27h28l-3-5H5l-3 5zM12 27h8"/>',
    '<path d="M9 15l14-8M9 17l14 8"/><circle cx="6" cy="16" r="4"/><circle cx="26" cy="5" r="4"/><circle cx="26" cy="27" r="4"/>',
]
CARDS = [
    ('1. Export', 'Export your trained model', 'Create prediction app files', 'automl.app()'),
    ('2. Run locally', 'Predict one row or a CSV', 'Open the app on your computer', 'automl.local_app()'),
    ('3. Share', 'Let others use your model', 'Publish online or host it yourself', 'automl.publish_app()'),
]


def render(output):
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="218" viewBox="0 0 1000 218" role="img">',
             '<title>Export, run locally, and share a prediction app</title>',
             '<desc>Generate app files from a trained model, run single-row or CSV predictions locally, then publish or host the app when ready.</desc>',
             '<g font-family="Arial, Helvetica, sans-serif">']
    for i, ((heading, action, description, command), icon) in enumerate(zip(CARDS, ICONS)):
        x = 2 + i*336
        parts.extend([
            f'<rect x="{x}" y="2" width="324" height="212" rx="14" fill="#ffffff" stroke="#cbdfe9"/>',
            f'<rect x="{x+18}" y="20" width="44" height="44" rx="12" fill="#eaf6fb"/>',
            f'<g transform="translate({x+24},26)" fill="none" stroke="#0086b3" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">{icon}</g>',
            f'<text x="{x+76}" y="48" font-size="20" font-weight="700" fill="#004666">{escape(heading)}</text>',
            f'<text x="{x+20}" y="102" font-size="18" fill="#004666">{escape(action)}</text>',
            f'<text x="{x+20}" y="132" font-size="16" fill="#52697a">{escape(description)}</text>',
            f'<rect x="{x+20}" y="160" width="{len(command)*8.5+28}" height="29" rx="7" fill="#eef4f8"/>',
            f'<text x="{x+34}" y="180" font-family="monospace" font-size="14" fill="#36566a">{escape(command)}</text>',
        ])
    parts.append('</g></svg>')
    Path(output).write_text('\n'.join(parts)+'\n')


if __name__ == '__main__':
    render(Path(__file__).resolve().parents[2] / 'docs/docs/assets/readme/app-workflow.svg')
