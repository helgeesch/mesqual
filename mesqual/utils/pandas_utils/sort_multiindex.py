import pandas as pd


def sort_multiindex(df: pd.DataFrame, custom_order: list[str | int], level: str | int, axis: int = 0) -> pd.DataFrame:
    """Sort a DataFrame's MultiIndex at a specific level using a custom order.

    Reorders the specified level according to custom_order while preserving
    the existing order of all other levels. This allows for sequential sorting
    operations where each sort maintains previous orderings.

    Values in the target level that are not included in custom_order will be
    appended at the end, maintaining their original relative order.

    Args:
        df: The DataFrame with MultiIndex to sort.
        custom_order: List of values defining the desired order for the specified level.
        level: Level to sort. Can be level name (str) or level number (int).
        axis: Axis to sort along. 0 for index (rows), 1 for columns.

    Returns:
        DataFrame with reordered MultiIndex according to the custom order.

    Raises:
        ValueError: If axis is not 0 or 1, or if level is not a valid string or integer.

    Example:

        >>> idx = pd.MultiIndex.from_arrays([['A', 'A', 'B', 'B'], [1, 2, 1, 2]])
        >>> df = pd.DataFrame({'val': [10, 20, 30, 40]}, index=idx)
        >>> sort_multiindex(df, [2, 1], level=1)  # Sort second level
    """
    if axis not in [0, 1]:
        raise ValueError("axis must be 0 (rows) or 1 (columns)")

    idx = df.axes[axis]
    if isinstance(idx, pd.MultiIndex):
        if isinstance(level, str):
            level_num = idx.names.index(level)
        elif isinstance(level, int):
            level_num = level
        else:
            raise ValueError("level must be a string (level name) or an integer (level number)")

        idx_tuples = idx.to_list()
        remaining_values_in_level_to_sort = [i for i in idx.get_level_values(level_num).unique() if
                                             i not in custom_order]
        ordered_tuples = []
        for value in custom_order + remaining_values_in_level_to_sort:
            for tuple_item in idx_tuples:
                if tuple_item[level_num] == value:
                    ordered_tuples.append(tuple_item)
        new_index = pd.MultiIndex.from_tuples(ordered_tuples, names=idx.names)
    else:
        idx_values = idx.to_list()
        remaining_values = [i for i in idx.unique() if i not in custom_order]
        ordered_values = []
        for value in custom_order + remaining_values:
            for idx_value in idx_values:
                if idx_value == value:
                    ordered_values.append(idx_value)
        new_index = pd.Index(ordered_values, name=idx.name)

    if axis == 0:
        return df.reindex(new_index)
    else:
        return df.reindex(columns=new_index)


if __name__ == '__main__':
    # Example 1: Basic MultiIndex sorting by level name
    arrays = [
        ['B', 'B', 'A', 'A', 'C', 'C'],
        ['x', 'y', 'x', 'y', 'x', 'y']
    ]
    index = pd.MultiIndex.from_arrays(arrays, names=('letter', 'symbol'))
    df = pd.DataFrame({'val': range(6)}, index=index)

    print("Example 1: Basic MultiIndex sorting")
    print("Original DataFrame:")
    print(df)
    print("\nAfter sorting 'letter' level with order ['C', 'A', 'B']:")
    df_sorted = sort_multiindex(df, ['C', 'A', 'B'], 'letter')
    print(df_sorted)

    # Example 2: Single-level index sorting
    print("\n" + "="*50)
    print("Example 2: Single-level index sorting")
    df_single = pd.DataFrame({'val': [10, 20, 30]}, index=['gamma', 'alpha', 'beta'])
    print("Original DataFrame:")
    print(df_single)
    print("\nAfter sorting with custom order ['beta', 'gamma', 'alpha']:")
    df_single_sorted = sort_multiindex(df_single, ['beta', 'gamma', 'alpha'], level=0)
    print(df_single_sorted)