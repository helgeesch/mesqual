import pandas as pd


def _tuple_sort(tuples: list[tuple], order: list, level: int) -> list[tuple]:
    def _sort_key(t: tuple) -> int:
        return order.index(t[level])
    result = []
    for group in {t[:level] + t[level + 1:] for t in tuples}:
        group_tuples = [t for t in tuples if t[:level] + t[level + 1:] == group]
        result.extend(sorted(group_tuples, key=_sort_key))
    return result


def sort_multiindex(df: pd.DataFrame, custom_order: list[str | int], level: str | int, axis=0) -> pd.DataFrame:
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
    idx = df.axes[axis]
    sorted_idx = _tuple_sort(idx.to_list(), custom_order, level_num)
    return df.reindex(sorted_idx, axis=axis)


if __name__ == '__main__':
    arrays = [['A', 'A', 'A', 'B', 'B', 'B'], ['x', 'y', 'z', 'x', 'y', 'z']]
    index = pd.MultiIndex.from_arrays(arrays, names=('letter', 'symbol'))
    df = pd.DataFrame({'val': [1, 2, 3, 4, 5, 6]}, index=index)

    print(sort_multiindex(df, ['z', 'x', 'y'], 'symbol'))