import os


def construct_learner_name(fold, repeat, repeats):
    repeat_str = f"_repeat_{repeat}" if repeats > 1 else ""
    return f"learner_fold_{fold}{repeat_str}"


def learner_name_to_fold_repeat(name):
    fold, repeat = None, None
    arr = name.split("_")
    fold = int(arr[2])
    if "repeat" in name:
        repeat = int(arr[4])
    return fold, repeat


def get_fold_repeat_cnt(model_path):
    training_logs = [f for f in os.listdir(model_path) if "_training.log" in f]
    fold_cnt, repeat_cnt = 0, 0
    for fname in training_logs:
        fold, repeat = learner_name_to_fold_repeat(fname)
        if fold is not None:
            fold_cnt = max(fold_cnt, fold)
        if repeat is not None:
            repeat_cnt = max(repeat_cnt, repeat)

    fold_cnt += 1  # counting from 0
    repeat_cnt += 1

    return fold_cnt, repeat_cnt


def get_learners_names(model_path):
    postfix = "_training.log"
    learner_names = [
        f.repleace(postfix, "") for f in os.listdir(model_path) if postfix in f
    ]
    return learner_names
