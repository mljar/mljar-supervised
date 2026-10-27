"""Render README figures from recorded runs; never reruns training.

Usage: python examples/readme/render_media.py performance-results/readme-visual-b1
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('run', type=Path)
args = parser.parse_args()
root = Path(__file__).resolve().parents[2]
out = root / 'docs/docs/assets/readme'
out.mkdir(parents=True, exist_ok=True)
measurements = json.loads((args.run / 'measurements.json').read_text())
plt.rcParams.update({'font.size': 13, 'font.family': 'DejaVu Sans',
                     'text.color': '#004666', 'axes.labelcolor': '#004666',
                     'axes.spines.top': False, 'axes.spines.right': False})

fairness = measurements['fairness']
fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
for ax, key, title in zip(axes, ['accuracy', 'demographic_parity_ratio'],
                        ['Test accuracy', 'Demographic parity ratio']):
    values = [fairness[n][key] for n in ['standard', 'mitigation']]
    ax.bar([0, 1], values, color=['#004666', '#0099cc'], width=.55)
    ax.set_xticks([0, 1], ['Standard', 'With mitigation'])
    ax.set_ylim(0, 1.05)
    ax.set_title(title, loc='left', fontweight='bold', pad=18)
    for x, value in enumerate(values):
        ax.text(x, value + .025, f'{value:.3f}', ha='center', fontsize=14)
    ax.spines[['left', 'bottom']].set_color('#bfd7ea')
    ax.grid(axis='y', alpha=.15)
    ax.set_axisbelow(True)
axes[1].axhline(.8, color='#705139', ls='--', lw=1.5)
axes[1].text(.5, .82, 'Chosen threshold: 0.80', ha='center', fontsize=10, color='#705139')
fig.subplots_adjust(bottom=.23, top=.85, wspace=.3)
fig.text(.5, .04, 'Synthetic data · same 200 test rows · MLJAR 1.4.0b1', ha='center', fontsize=11)
fig.savefig(out / 'fairness-comparison.png', dpi=160, facecolor='white')
plt.close(fig)

lb = pd.read_csv(args.run / 'feature_engineering/leaderboard.csv')
selected = lb.set_index('name').loc[['1_Linear', '1_Linear_GoldenFeatures', '5_Linear_FeatureDiscovery_Compact']]
fig, ax = plt.subplots(figsize=(10, 4.2))
values = selected.metric_value.to_numpy()
ax.barh([2, 1, 0], values, color=['#004666', '#0099cc', '#528244'], height=.55)
ax.set_yticks([2, 1, 0], ['Original columns', 'Golden features', 'Feature discovery'])
ax.set_xlim(0, max(values)*1.22)
for row, value in zip([2, 1, 0], values):
    ax.text(value + .06, row, f'{value:.3f}', va='center')
ax.set_xlabel('Validation RMSE · lower is better')
ax.set_title('Additional features, evaluated with the same Linear model', loc='left', fontsize=15, fontweight='bold', pad=20)
ax.spines[['left', 'bottom']].set_color('#bfd7ea')
fig.subplots_adjust(left=.22, right=.96, bottom=.25, top=.81)
fig.text(.5, .045, 'Synthetic length × width example · one shared validation split · MLJAR 1.4.0b1', ha='center', fontsize=10)
fig.savefig(out / 'feature-comparison.png', dpi=160, facecolor='white')
plt.close(fig)

# Code-native illustrations are capability diagrams, not product screenshots.
from render_model_families import render as render_model_families
from render_app_workflow import render as render_app_workflow

render_model_families(out / 'model-families.svg')
render_app_workflow(out / 'app-workflow.svg')
