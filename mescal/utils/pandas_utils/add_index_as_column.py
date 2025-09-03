import pandas as pd


def add_index_as_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds the DataFrame's index to a new column while preserving the original index.

    This function handles single-level, unnamed, and multi-level indices.
    - For a single-level index, a new column with the index's name is created.
      If the index is unnamed, the new column will be named 'index'.
    - For a MultiIndex, a new column is created for each index level, and
      an additional column named 'combined_index' holds the full index tuple.

    Args:
        df: The pandas DataFrame to process.

    Returns:
        A new DataFrame with the index added as a column. The original DataFrame
        is not modified.
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


if __name__ == '__main__':

    # Example 1: DataFrame with a named index
    print("--- Example 1: Named Index ---")
    data = {'A': [10, 20, 30], 'B': [1, 2, 3]}
    df_named = pd.DataFrame(data).set_index(pd.Index(['X', 'Y', 'Z'], name='label'))

    print("Original DataFrame (Named Index):")
    print(df_named)
    print("\nDataFrame with index as column:")
    df_named_result = add_index_as_column(df_named)
    print(df_named_result)
    print("-" * 40)

    # Example 2: DataFrame with an unnamed index
    print("\n--- Example 2: Unnamed Index ---")
    data = {'C': [100, 200, 300], 'D': [10, 20, 30]}
    df_unnamed = pd.DataFrame(data)

    print("Original DataFrame (Unnamed Index):")
    print(df_unnamed)
    print("\nDataFrame with index as column:")
    df_unnamed_result = add_index_as_column(df_unnamed)
    print(df_unnamed_result)
    print("-" * 40)

    # Example 3: DataFrame with a MultiIndex
    print("\n--- Example 3: MultiIndex ---")
    data = {
        'value': [1000, 2000, 3000, 4000],
        'category': ['A', 'A', 'B', 'B'],
        'item_id': [1, 2, 1, 2]
    }
    df_multi = pd.DataFrame(data).set_index(['category', 'item_id'])

    print("Original DataFrame (MultiIndex):")
    print(df_multi)
    print("\nDataFrame with index as column:")
    df_multi_result = add_index_as_column(df_multi)
    print(df_multi_result)
    print("-" * 40)
