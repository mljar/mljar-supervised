"""Legacy callback metadata; scheduling is enforced at fold boundaries.

ModelFramework and FoldBudget own the stopping decisions, including finishing
reserve and candidate completion tracking. This callback never truncates a fit.
"""

from supervised.callbacks.callback import Callback


class TotalTimeConstraint(Callback):
    def __init__(self, params={}):
        super(TotalTimeConstraint, self).__init__(params)
        self.name = params.get("name", "total_time_constraint")
        self.total_time_limit = params.get("total_time_limit")
        self.total_time_start = params.get("total_time_start")
        self.expected_learners_cnt = params.get("expected_learners_cnt", 1)
