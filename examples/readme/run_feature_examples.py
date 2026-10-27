"""Produce small, synthetic feature-discovery and fairness examples for the README.

Run from a new output directory with mljar-supervised==1.4.0b1 installed.
Synthetic rows illustrate the APIs, not performance on a real application.
"""
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from supervised import AutoML, __version__

rng = np.random.default_rng(42)
X = pd.DataFrame({
    'length': rng.uniform(1, 10, 500),
    'width': rng.uniform(1, 5, 500),
    'material': rng.choice(['A', 'B', 'C'], 500),
})
y = X.length * X.width + (X.material == 'B') * 4 + rng.normal(0, 0.5, 500)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
model = AutoML(
    results_path='feature_engineering', mode='Compete', algorithms=['Linear'],
    total_time_limit=120, n_jobs=2, explain_level=0,
    golden_features=True, feature_discovery=True,
    features_selection=False, kmeans_features=False, mix_encoding=False,
    boost_on_errors=False, train_ensemble=False, stack_models=False,
    start_random_models=1, hill_climbing_steps=0,
    validation_strategy={'validation_type': 'split', 'train_ratio': 0.8, 'shuffle': True, 'stratify': False},
)
model.fit(X_train, y_train)
feature_metrics = {'test_mae': mean_absolute_error(y_test, model.predict(X_test))}
model.report()

# A synthetic proxy correlates with the group and the observed target.
# The groups have no demographic meaning. This is only a mechanics example.
group = rng.integers(0, 2, 800)
signal = rng.normal(size=800)
X = pd.DataFrame({'signal': signal, 'proxy': group + rng.normal(0, 0.25, 800)})
y = pd.Series((signal + 1.5 * group + rng.normal(0, 0.7, 800) > 0.75).astype(int))
S = pd.DataFrame({'group': np.where(group, 'B', 'A')})
X_train, X_test, y_train, y_test, S_train, S_test = train_test_split(
    X, y, S, test_size=0.25, stratify=y, random_state=42
)
fairness_metrics = {}
for name, extra in [
    ('standard', {}),
    ('mitigation', {'fairness_metric': 'demographic_parity_ratio', 'fairness_threshold': 0.8}),
]:
    model = AutoML(
        results_path=name, mode='Explain', algorithms=['Linear'],
        total_time_limit=60, n_jobs=2, explain_level=0, train_ensemble=False,
        **extra,
    )
    model.fit(X_train, y_train, **({'sensitive_features': S_train} if extra else {}))
    predictions = model.predict(X_test).astype(int)
    rates = {g: float(predictions[S_test.group.to_numpy() == g].mean()) for g in ['A', 'B']}
    fairness_metrics[name] = {
        'accuracy': accuracy_score(y_test, predictions), 'selection_rates': rates,
        'demographic_parity_ratio': min(rates.values()) / max(rates.values()),
        'test_rows': len(y_test), 'threshold': 'AutoML saved decision threshold',
    }
    model.report()

Path('measurements.json').write_text(json.dumps({
    'mljar': __version__, 'python': platform.python_version(), 'seed': 42,
    'data': 'synthetic; see run_feature_examples.py',
    'feature_engineering': feature_metrics, 'fairness': fairness_metrics,
}, indent=2) + '\n')
