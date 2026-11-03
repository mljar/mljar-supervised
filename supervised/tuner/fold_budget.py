"""Budget decisions at completed-fold boundaries, never inside a fit."""

import time

import numpy as np

from supervised.exceptions import BudgetExceededError


class FoldBudget:
    def __init__(self, available=None, finishing_available=None, allowance=None, notify=None):
        self.started = time.monotonic()
        self.available = available
        self.finishing_available = available if finishing_available is None else finishing_available
        self.allowance = allowance
        self.folds = []
        self.decisions = []
        self.status = "not_started"
        self.notify = notify
        self.reason = None

    def check(self, expected_folds):
        completed = len(self.folds)
        if not completed or completed == expected_folds or self.available is None:
            return
        elapsed = time.monotonic() - self.started
        costs = [f["seconds"] for f in self.folds]
        # One-time candidate setup is in elapsed, but is not multiplied by folds.
        typical = float(np.median(costs))
        margin = max(0.2 * typical, float(np.std(costs)))
        # Reserve one typical fold's worth for aggregation, selection and reports.
        remaining_cost = (typical + margin) * (expected_folds - completed) + typical
        finishing = completed / expected_folds >= 0.8
        available = (self.finishing_available if finishing else self.available) - elapsed
        decision = dict(
            completed_folds=completed, expected_folds=expected_folds,
            elapsed=elapsed, estimated_remaining=remaining_cost,
            estimated_total=elapsed + remaining_cost, available_remaining=available,
            finishing_reserve=finishing,
        )
        if remaining_cost > available:
            self.decisions.append(dict(decision, decision="stop"))
            raise BudgetExceededError(
                f"Not trained: completed {completed}/{expected_folds} folds; "
                f"estimated remaining work {remaining_cost:.1f}s exceeds "
                f"available time {max(0, available):.1f}s.",
                phase="between_folds", **decision,
            )
        estimated = elapsed + remaining_cost
        extended = self.allowance is not None and estimated > self.allowance
        if extended:
            self.allowance = estimated
        self.decisions.append(dict(decision, decision="extend" if extended else "continue"))
        if self.notify is not None and extended:
            self.notify(
                f"Completed {completed}/{expected_folds} folds; estimated remaining "
                f"work {remaining_cost:.1f}s, available {available:.1f}s. "
                "Extending candidate allocation."
            )

    def record(self, learner, seconds, stop_reason):
        self.folds.append(dict(learner=learner, seconds=seconds, stop_reason=stop_reason))

    def report(self, expected_folds):
        return dict(status=self.status, completed_folds=len(self.folds),
                    expected_folds=expected_folds, folds=self.folds, decisions=self.decisions,
                    candidate_allowance=self.allowance, reason=self.reason)
