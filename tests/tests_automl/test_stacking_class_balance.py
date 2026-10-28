import json
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from supervised import AutoML
from supervised.algorithms.registry import BINARY_CLASSIFICATION, MULTICLASS_CLASSIFICATION, REGRESSION


@pytest.mark.parametrize('counts,task,expected', [
    ([86, 14], BINARY_CLASSIFICATION, False),
    ([85, 15], BINARY_CLASSIFICATION, True),
    ([84, 16], BINARY_CLASSIFICATION, True),
    ([70, 16, 14], MULTICLASS_CLASSIFICATION, False),
    ([70, 15, 15], MULTICLASS_CLASSIFICATION, True),
    ([86, 14], REGRESSION, True),
])
def test_class_share_boundary(counts, task, expected):
    model = AutoML()
    model._ml_task = task
    model._stack_models = True
    model.verbose_print = Mock()
    y = pd.Series(np.repeat(np.arange(len(counts)), counts))
    model._apply_class_balance_stacking_constraint(y)
    assert model._stack_models is expected
    if not expected:
        assert 'below 15%' in model.verbose_print.call_args.args[0]
    else:
        model.verbose_print.assert_not_called()


def test_unused_categories_do_not_disable_stacking():
    model = AutoML()
    model._ml_task = BINARY_CLASSIFICATION
    model._stack_models = True
    y = pd.Series(pd.Categorical(['a'] * 85 + ['b'] * 15, categories=['a', 'b', 'unused']))
    model._apply_class_balance_stacking_constraint(y)
    assert model._stack_models


def test_disabled_stacking_is_not_reenabled():
    model = AutoML()
    model._ml_task = BINARY_CLASSIFICATION
    model._stack_models = False
    model._apply_class_balance_stacking_constraint(pd.Series([0, 1]))
    assert not model._stack_models


@pytest.mark.parametrize('validation', ['auto', {'validation_type': 'kfold', 'k_folds': 2}])
@pytest.mark.parametrize('stack_models', ['auto', True])
def test_fit_disables_stacking_but_keeps_plain_ensemble(tmp_path, capsys, validation, stack_models):
    # >500 rows isolates the class-balance policy from the old small-data rule.
    y = np.array(['majority'] * 516 + ['minority'] * 84)
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(600, 3)))
    model = AutoML(
        mode='Compete', algorithms=['Decision Tree'], results_path=str(tmp_path / 'run'),
        total_time_limit=60, stack_models=stack_models, validation_strategy=validation,
        start_random_models=1, hill_climbing_steps=0, train_ensemble=True,
        golden_features=False, feature_discovery=False, features_selection=False,
        kmeans_features=False, mix_encoding=False, boost_on_errors=False,
        explain_level=0, n_jobs=1, verbose=1,
    )
    model.fit(X, y)
    assert not model._stack_models
    assert not model.tuner._stack_models
    assert not model._time_ctrl._is_stacking
    assert 'stack' not in model._time_ctrl._steps
    assert model._train_ensemble
    assert all(not m._is_stacked for m in model._models)
    params = json.loads((tmp_path / 'run/params.json').read_text())
    assert params['stack_models'] is False
    assert '84/600 training rows (14.00%), below 15%' in capsys.readouterr().out
    assert len(model.predict(X[:3])) == 3


def test_class_share_uses_clean_targets_not_sample_weights(tmp_path, capsys):
    rng = np.random.default_rng(17)
    X = pd.DataFrame(rng.normal(size=(700, 3)))
    y = pd.Series(["majority"] * 510 + ["minority"] * 90 + [None] * 100)
    weights = np.array([1.0] * 510 + [100.0] * 90 + [1.0] * 100)
    model = AutoML(
        mode="Compete", algorithms=["Decision Tree"],
        results_path=str(tmp_path / "clean-targets"), total_time_limit=60,
        stack_models=True, validation_strategy={"validation_type": "kfold", "k_folds": 2},
        start_random_models=1, hill_climbing_steps=0, train_ensemble=False,
        golden_features=False, feature_discovery=False, features_selection=False,
        kmeans_features=False, mix_encoding=False, boost_on_errors=False,
        explain_level=0, n_jobs=1,
    )
    model.fit(X, y, sample_weight=weights)
    # Exactly 15% after missing-target removal. Including the missing rows or
    # weighting the counts would incorrectly reject this balanced-enough target.
    assert model.n_rows_in_ == 600
    assert model._stack_models
    assert model.tuner._stack_models
    assert "below 15%" not in capsys.readouterr().out
