"""Prediction reuse for one input batch and its ensemble dependencies."""


class PredictionContext:
    """Keep intermediate predictions only for the lifetime of one request.

    Inputs are identified by object identity, not just model name or row count:
    original and stacked frames must never share an entry. Entries retain their
    inputs and model so Python cannot reuse their identities during the request.
    Callers must not mutate cached frames (copy before changing labels/columns).
    """

    def __init__(self):
        self._predictions = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._predictions.clear()

    def predict(self, model, X, X_stacked=None):
        key = (id(model), id(X), id(X_stacked))
        if key not in self._predictions:
            if model.get_type() == "Ensemble":
                prediction = model.predict(X, X_stacked, prediction_context=self)
            else:
                prediction = model.predict(X)
            self._predictions[key] = (model, X, X_stacked, prediction)
        return self._predictions[key][-1]
