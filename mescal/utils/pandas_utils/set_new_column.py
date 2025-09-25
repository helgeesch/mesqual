from typing import Hashable
import pandas as pd


def set_column(
        df: pd.DataFrame,
        new_column_name: Hashable,
        new_column_values: pd.Series | pd.DataFrame
) -> pd.DataFrame:
    """Set or replace a column in a DataFrame with new values.

    Adds a new column or replaces an existing column in a DataFrame. Handles both
    Series and DataFrame inputs, with special logic for MultiIndex columns when
    using DataFrame inputs.

    Args:
        df: The DataFrame to modify.
        new_column_name: Name/key for the new column.
        new_column_values: Values for the new column. Can be a Series for simple
            columns or a DataFrame for MultiIndex column structures.

    Returns:
        A copy of the DataFrame with the new column added or existing column replaced.

    Raises:
        ValueError: If length of df and new_column_values don't match, or if
            new_column_values DataFrame has incorrect number of column levels.
        TypeError: If new_column_values is neither Series nor DataFrame.

    Examples:

        >>> import pandas as pd
        >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        >>>
        >>> # Add Series as new column
        >>> new_series = pd.Series([7, 8, 9])
        >>> result = set_column(df, 'C', new_series)
        >>> print(result.columns.tolist())
            ['A', 'B', 'C']
        >>>
        >>> # Replace existing column
        >>> replacement = pd.Series([10, 11, 12])
        >>> result = set_column(df, 'A', replacement)
        >>> print(result['A'].tolist())
            [10, 11, 12]
    """

    dff = df.copy()

    if not len(dff) == len(new_column_values):
        raise ValueError('Length of dff and new_column_values must be equal.')

    # TODO optional: check index

    if isinstance(new_column_values, pd.Series):
        dff[new_column_name] = new_column_values
        return dff

    if isinstance(new_column_values, pd.DataFrame):
        if not new_column_values.columns.nlevels == (dff.columns.nlevels - 1):
            raise ValueError(
                'Your new_column_values must have n-1 column levels, where n is the number of levels in dff.'
            )

        if new_column_name in dff.columns:
            dff = dff.drop(columns=[new_column_name])

        new_column_values = pd.concat({new_column_name: new_column_values}, axis=1, names=[dff.columns.names[0]])
        dff = pd.concat([dff, new_column_values], axis=1)
        return dff

    else:
        raise TypeError('Used new_column_values type not accepted.')


if __name__ == '__main__':
    # Example 1: Adding a Series as a new column
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    new_series = pd.Series([7, 8, 9])
    result1 = set_column(df, 'C', new_series)
    print("Example 1 - Adding Series column:")
    print(result1)
    print()

    # Example 2: Replacing an existing column
    replacement = pd.Series([10, 11, 12])
    result2 = set_column(df, 'A', replacement)
    print("Example 2 - Replacing existing column:")
    print(result2)
    print()

    # Example 3: Adding DataFrame to MultiIndex columns
    multi_df = pd.DataFrame(
        [[1, 2], [3, 4], [5, 6]],
        columns=pd.MultiIndex.from_tuples([('X', 'a'), ('X', 'b')], names=['level1', 'level2'])
    )
    new_df_values = pd.DataFrame([[7], [8], [9]], columns=['c'])
    result3 = set_column(multi_df, 'Y', new_df_values)
    print("Example 3 - Adding DataFrame to MultiIndex columns:")
    print(result3)