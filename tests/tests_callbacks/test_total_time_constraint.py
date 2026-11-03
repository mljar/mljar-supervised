from types import SimpleNamespace

from supervised.callbacks.total_time_constraint import TotalTimeConstraint


def test_callback_never_interrupts_an_admitted_fold():
    callback = TotalTimeConstraint(dict(total_time_limit=0, total_time_start=0,
                                       expected_learners_cnt=15))
    learner = SimpleNamespace(stop_training=False)
    callback.add_and_set_learner(learner)
    callback.on_learner_train_start(None)
    callback.on_iteration_end(None, None)
    callback.on_learner_train_end(None)
    assert not learner.stop_training
