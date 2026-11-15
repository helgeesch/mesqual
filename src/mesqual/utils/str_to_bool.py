"""Conversion of strings and other values to bool."""


def any_to_bool(value) -> bool:
    """Convert a value to a bool: a string with `str_to_bool`, any other value by its truth value.

    So `None` gives False, and NaN gives True.

    Args:
        value (Any): The value.
    """
    if isinstance(value, str):
        return str_to_bool(value)
    elif value:
        return True
    return False


def str_to_bool(text: str) -> bool:
    """Return False for `false`, `f`, `0`, `no`, `n`, `off` or an empty string, in any case; True for any other string."""
    return str(text).lower() not in ['false', 'f', '0', 'no', 'n', 'off', '']
