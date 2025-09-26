def any_to_bool(value) -> bool:
    if isinstance(value, str):
        return str_to_bool(value)
    elif value:
        return True
    return False


def str_to_bool(text: str) -> bool:
    """
    Converts string to boolean, accepting following representations:
    For False: ['False', 'false', 'F', 'f', '0', 'no', 'n', 'N', 'No', 'NO', '']
    For True : ['True', 'true', 'T', 't', '1', 'yes', 'y', 'Y', 'Yes', 'YES']
    """
    return str(text).lower() not in ['false', 'f', '0', 'no', 'n', 'off', '']
