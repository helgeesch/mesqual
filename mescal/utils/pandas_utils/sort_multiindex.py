import pandas as pd


def sort_multiindex(df: pd.DataFrame, custom_order: list[str | int], level: str | int, axis=0):
    """
    Sorts a DataFrame's MultiIndex at a specific level according to a custom order,
    while preserving the order of other levels.
    """
    if axis not in [0, 1]:
        raise ValueError("axis must be 0 (rows) or 1 (columns)")

    if isinstance(level, str):
        level_num = df.axes[axis].names.index(level)
    elif isinstance(level, int):
        level_num = level
    else:
        raise ValueError("level must be a string (level name) or an integer (level number)")

    def _custom_sort_key(idx_value):
        if idx_value[level_num] in custom_order:
            return custom_order.index(idx_value[level_num])
        else:
            return float('inf')

    sorted_idx = sorted(df.axes[axis], key=_custom_sort_key)

    return df.reindex(sorted_idx, axis=axis)
