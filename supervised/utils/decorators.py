from functools import wraps

from supervised.exceptions import AutoMLException


def fit_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if f.is_fitted() == False:
            raise AutoMLException(
                "Model has not yeet been fitted. Please call `fit()` first."
            )
        return f(*args, **kwargs)

    return decorated


def no_fit_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if f.is_fitted() == True:
            raise AutoMLException(
                f"Cannot use method `{f.__name__}` after AutoML has been fitted`."
            )
        return f(*args, **kwargs)

    return decorated


def check_greater_than_zero_integer(value, original_var_name):
    if not isinstance(value, int):
        raise ValueError(f"{original_var_name} must be an integer, got {type(value)}.")

    if value <= 0:
        raise ValueError(f"{original_var_name} must be greater than zero, got {value}.")


def check_positive_integer(value, original_var_name):
    if not isinstance(value, int):
        raise ValueError(f"{original_var_name} must be an integer, got {type(value)}.")

    if value < 0:
        raise ValueError(
            f"{original_var_name} must be equal or greater than zero, got {value}."
        )


def check_bool(value, original_var_name):
    if not isinstance(value, bool):
        raise ValueError(f"{original_var_name} must be a boolean, got {type(value)}.")
