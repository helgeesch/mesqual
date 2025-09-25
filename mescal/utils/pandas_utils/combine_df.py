from typing import Iterable
from functools import reduce

import pandas as pd


def combine_dfs(dfs: Iterable[pd.Series | pd.DataFrame], keep_first: bool = True):
    """Combine multiple DataFrames or Series using intelligent merging strategies.

    This function automatically determines how to combine DataFrames based on their
    index and column structure:
    - If DataFrames share indices but not columns: concatenate along columns (axis=1)
    - If DataFrames share columns but not indices: concatenate along rows (axis=0)
    - Otherwise: use combine_first() to fill missing values

    Args:
        dfs: An iterable of pandas DataFrames or Series to combine.
        keep_first: If True, prioritize values from earlier DataFrames when using
            combine_first(). If False, prioritize values from later DataFrames.

    Returns:
        A single DataFrame or Series containing the combined data.

    Raises:
        ValueError: If no DataFrames are provided in the iterable.

    Energy Domain Context:
        In Energy Systems Analysis, you often deal with fragmented data,
        stored or imported from different locations. For example:

            - You have multiple simulation results that you want to concatenate,
                e.g. one each model covers only one month and you
                need to merge those into a single df
            - You have a yearly simulation result, but one week must be
                replaced with another result, because you had to re-run
                that with a different setting. So only that week should
                be overwritten.
            - You have a local csv file with static properties that you
                want to merge with the model_df that is coming from the
                simulation platform.

    Examples:

        >>> import pandas as pd
        >>> df1 = pd.DataFrame({'A': [1, 2]}, index=['x', 'y'])
        >>> df2 = pd.DataFrame({'B': [3, 4]}, index=['x', 'y'])
        >>> result = combine_dfs([df1, df2])  # Column concat
        >>> print(result)
               A  B
            x  1  3
            y  2  4

        >>> df3 = pd.DataFrame({'A': [5, 6]}, index=['z', 'w'])
        >>> result = combine_dfs([df1, df3])  # Row concat
        >>> print(result)
               A
            x  1
            y  2
            z  5
            w  6
    """
    size = sum(1 for _ in dfs)
    if size == 0:
        raise ValueError("You need to pass at least one DataFrame / Series.")
    if size == 1:
        return [df for df in dfs][0]

    def merge_func(df1, df2):
        if isinstance(df1, pd.DataFrame) and isinstance(df2, pd.DataFrame):
            column_intersection = set(df1.columns).intersection(df2.columns)
            index_intersection = set(df1.index).intersection(df2.index)
            if (len(column_intersection) == 0) and (len(index_intersection) > 0):
                return pd.concat([df1, df2], axis=1)
            elif (len(column_intersection) > 0) and (len(index_intersection) == 0):
                return pd.concat([df1, df2], axis=0)

        if keep_first:
            return df1.combine_first(df2)
        else:
            return df2.combine_first(df1)

    return reduce(merge_func, dfs)


if __name__ == "__main__":
    # Example 1: Column concatenation (shared indices, different columns)
    df1 = pd.DataFrame({'A': [1, 2, 3]}, index=['x', 'y', 'z'])
    df2 = pd.DataFrame({'B': [4, 5, 6]}, index=['x', 'y', 'z'])

    result1 = combine_dfs([df1, df2])
    print("Example 1 - Column concatenation:")
    print(result1)
    print()

    # Example 2: Row concatenation (different indices, shared columns)
    df3 = pd.DataFrame({'A': [10, 20]}, index=['w', 'v'])

    result2 = combine_dfs([df1, df3])
    print("Example 2 - Row concatenation:")
    print(result2)
    print()

    # Example 3: combine_first behavior with overlapping data
    df4 = pd.DataFrame({'A': [None, 100, None]}, index=['x', 'y', 'z'])
    df5 = pd.DataFrame({'A': [200, None, 300]}, index=['x', 'y', 'z'])

    result3 = combine_dfs([df4, df5], keep_first=True)
    print("Example 3 - combine_first (keep_first=True):")
    print(result3)
