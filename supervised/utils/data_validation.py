def check_greater_than_zero_integer(value, original_var_name):
    if not isinstance(value, int):
        raise ValueError(
            f"'{original_var_name}' must be an integer, got '{type(value)}'."
        )

    if value <= 0:
        raise ValueError(
            f"'{original_var_name}' must be greater than zero, got '{value}'."
        )


def check_positive_integer(value, original_var_name):
    if not isinstance(value, int):
        raise ValueError(
            f"'{original_var_name}' must be an integer, got '{type(value)}'."
        )

    if value < 0:
        raise ValueError(
            f"'{original_var_name}' must be equal or greater than zero, got '{value}'."
        )


def check_integer(value, original_var_name):
    if not isinstance(value, int):
        raise ValueError(
            f"'{original_var_name}' must be an integer, got '{type(value)}'."
        )


def check_bool(value, original_var_name):
    if not isinstance(value, bool):
        raise ValueError(
            f"'{original_var_name}' must be a boolean, got '{type(value)}'."
        )


def check_greater_than_zero_integer_or_float(value, original_var_name):

    if not (isinstance(value, int) or isinstance(value, float)):
        raise ValueError(
            f"'{original_var_name}' must be an integer or float, got '{type(value)}'."
        )

    if value <= 0:
        raise ValueError(
            f"'{original_var_name}' must be greater than zero, got '{value}'."
        )
