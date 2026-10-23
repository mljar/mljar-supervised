"""Bound the prediction dependencies introduced by a new stack."""

_MAX_MODELS_PER_TYPE = 5


def select_stacking_models(models, require_fair=False):
    # Loss is always minimized internally, including negative AUC. Break ties
    # by name so resume/order changes cannot change the generated column schema.
    eligible = [m for m in models if not m._is_stacked
                and m.get_type() not in {"Baseline", "Ensemble"}
                and "RandomFeature" not in m.get_name()
                and m.is_valid()
                and (not require_fair or m.is_fair())]
    counts, selected = {}, []
    for model in sorted(eligible, key=lambda m: (m.get_final_loss(), m.get_name())):
        kind = model.get_type()
        if counts.get(kind, 0) < _MAX_MODELS_PER_TYPE:
            selected.append(model)
            counts[kind] = counts.get(kind, 0) + 1
    return selected
