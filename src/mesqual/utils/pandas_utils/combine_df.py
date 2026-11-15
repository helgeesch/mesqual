from typing import Iterable, Literal
from functools import reduce

import pandas as pd


def combine_dfs(
    dfs: Iterable[pd.Series | pd.DataFrame],
    precedence: Literal['first', 'last'] = 'first',
) -> pd.Series | pd.DataFrame:
    """Combine multiple DataFrames or Series into one.

    The index and column structure of the inputs selects the method:

    - Shared index, no shared columns: concatenate along the columns.
    - Shared columns, no shared index: concatenate along the rows.
    - Otherwise: `combine_first`, so that one input fills the gaps of the other.

    Args:
        dfs: An iterable of pandas DataFrames or Series to combine.
        precedence: Which input wins where two inputs overlap in index and
            columns: `'first'` keeps the values of the earlier input,
            `'last'` those of the later input.

    Returns:
        A single DataFrame or Series containing the combined data.

    Raises:
        ValueError: If no DataFrames are provided in the iterable, or
            `precedence` is not `'first'` or `'last'`.

    Use it to join results that come in parts, for example one run per
    month, or a yearly run in which one week is replaced by a rerun.

    Examples:
        >>> first = pd.DataFrame({'A': [1, 2]}, index=['x', 'y'])
        >>> combine_dfs([first, pd.DataFrame({'B': [3, 4]}, index=['x', 'y'])])  # new columns
           A  B
        x  1  3
        y  2  4
        >>> combine_dfs([first, pd.DataFrame({'A': [5]}, index=['z'])])  # new rows
           A
        x  1
        y  2
        z  5
    """
    if precedence not in ('first', 'last'):
        raise ValueError(f"precedence must be 'first' or 'last', got {precedence!r}.")
    dfs = list(dfs)
    if len(dfs) == 0:
        raise ValueError('You need to pass at least one DataFrame / Series.')
    if len(dfs) == 1:
        return dfs[0]

    def merge_func(df1, df2):
        if isinstance(df1, pd.DataFrame) and isinstance(df2, pd.DataFrame):
            column_intersection = set(df1.columns).intersection(df2.columns)
            index_intersection = set(df1.index).intersection(df2.index)
            if (len(column_intersection) == 0) and (len(index_intersection) > 0):
                return pd.concat([df1, df2], axis=1)
            elif (len(column_intersection) > 0) and (len(index_intersection) == 0):
                return pd.concat([df1, df2], axis=0)

        if precedence == 'first':
            return df1.combine_first(df2)
        return df2.combine_first(df1)

    return reduce(merge_func, dfs)
