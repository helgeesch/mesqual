"""Add the index of a frame as a column."""

import pandas as pd


def add_index_as_column(df: pd.DataFrame) -> pd.DataFrame:
    """Adds the DataFrame's index to a new column while preserving the original index.

    This function handles single-level, unnamed, and multi-level indices:

    - For a single-level index, a new column with the index's name is created.
      If the index is unnamed, the new column will be named 'index'.
    - For a MultiIndex, a new column is created for each index level, and
      an additional column named 'combined_index' holds the full index tuple.

    Args:
        df: The pandas DataFrame to process.

    Returns:
        A new DataFrame with the index added as a column. The original DataFrame
            is not modified.

    Examples:
        A named index:

        >>> df = pd.DataFrame({'A': [10, 20]}, index=pd.Index(['X', 'Y'], name='label'))
        >>> add_index_as_column(df).columns.tolist()
        ['A', 'label']

        A MultiIndex:

        >>> df_multi = pd.DataFrame(
        ...     {'value': [100, 200]}, index=pd.MultiIndex.from_tuples([('A', 1), ('B', 2)], names=['cat', 'id']),
        ... )
        >>> add_index_as_column(df_multi).columns.tolist()
        ['value', 'cat', 'id', 'combined_index']
    """
    # Create a copy to avoid modifying the original DataFrame
    df_copy = df.copy()

    # Check if the index is a MultiIndex
    if isinstance(df_copy.index, pd.MultiIndex):
        # Get the names of the index levels
        level_names = df_copy.index.names

        # Add columns for each level of the MultiIndex
        for i, name in enumerate(level_names):
            if name not in df_copy.columns:
                df_copy[name] = df_copy.index.get_level_values(i)

        # Add a column for the combined index tuple
        df_copy['combined_index'] = df_copy.index.to_list()

    else:
        # Get the index name, defaulting to 'index' if unnamed
        index_name = df_copy.index.name
        if index_name is None:
            index_name = 'index'

        # Add the index as a new column
        if index_name not in df_copy.columns:
            df_copy[index_name] = df_copy.index

    return df_copy
