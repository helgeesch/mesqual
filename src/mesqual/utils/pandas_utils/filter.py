import pandas as pd

from mesqual.utils.pandas_utils.pend_props import get_matching_axis_and_level


def filter_by_model_query(
    df: pd.Series | pd.DataFrame,
    model_df: pd.DataFrame,
    query: str = None,
    match_on_level: int | str = None,
) -> pd.DataFrame | pd.Series:
    """Filter DataFrame or Series based on a query applied to a model DataFrame.

    This function filters data by applying a pandas query to a model DataFrame and
    using the resulting index to filter the target DataFrame or Series. It handles
    both simple and MultiIndex cases automatically.

    Args:
        df: The DataFrame or Series to filter.
        model_df: The model DataFrame containing metadata used for filtering.
            Must have an index that can be matched against df's axis.
        query: A pandas query string to apply to model_df. If None or empty,
            returns df unchanged. Uses pandas query syntax.
        match_on_level: For MultiIndex cases, specifies which level to match on.
            Can be an integer (level position) or string (level name).

    Returns:
        Filtered DataFrame or Series with the same type as input df.

    Examples:
        >>> generation = dataset.fetch('Generator.Results.generation')
        >>> generators = dataset.fetch('Generator.Model')
        >>> generators
                  node     fuel  capacity
        Generator
        G1          N1  nuclear     800.0
        G2          N1     coal     900.0
        G3          N2      gas    1000.0
        G4          N2     wind    1100.0
        G5          N3    solar    1200.0
        >>> renewable = filter_by_model_query(generation, generators, 'fuel in ["wind", "solar"]')
        >>> sorted(renewable.columns)
        ['G4', 'G5']
    """
    if query is None or query == '':
        return df

    axis, idx_selection_level = get_matching_axis_and_level(df, model_df.index, match_on_level)
    idx = df.axes[axis]

    selection = model_df.query(query, engine='python').copy(deep=True).index
    selection = list(set(selection).intersection(idx.get_level_values(idx_selection_level)))

    if isinstance(idx, pd.MultiIndex):
        selection = [i for i in idx if i[idx_selection_level] in selection]

    if isinstance(df, pd.Series):
        return df[selection] if axis == 0 else df.loc[selection]
    else:
        return df.loc[selection, :] if axis == 0 else df.loc[:, selection]
