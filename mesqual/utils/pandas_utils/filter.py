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

    Example:

        >>> # You have a generation time-series df
        >>> print(gen_df)  # Original DataFrame
            generator            GenA  GenB  GenC  SolarA  WindA
            2024-01-01 00:00:00   100   200   150      50     80
            2024-01-01 01:00:00   120   180   170      60     90
            2024-01-01 02:00:00   110   190   160      55     85

        >>> # You have a generator model df
        >>> print(model_df)
                      zone technology  is_res
            generator
            GenA        DE    nuclear   False
            GenB        DE       coal   False
            GenC        FR        gas   False
            SolarA      DE      solar    True
            WindA       NL       wind    True

        >>> only_de_conv = filter_by_model_query(gen_df, model_df, '(not is_res) and (zone == "DE")')
        >>> print(only_de_conv)  # DataFrame with only non-res generators in DE
            generator            GenA GenB
            2024-01-01 00:00:00   100  200
            2024-01-01 01:00:00   120  180
            2024-01-01 02:00:00   110  190
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


if __name__ == '__main__':
    import numpy as np

    # Set up model data (metadata about zones)
    model_data = {
        'zone': ['DE', 'FR', 'NL', 'BE'],
        'price': [50, 45, 55, 60],
        'has_renewable': [True, False, True, False]
    }
    model_df = pd.DataFrame(model_data).set_index('zone')
    print("Model DataFrame:")
    print(model_df)
    print()

    # Example 1: Filter DataFrame with MultiIndex columns
    print("=== Example 1: DataFrame with MultiIndex columns ===")
    timeindex = pd.date_range('2024-01-01', periods=3, freq='h')
    zones = ['DE', 'FR', 'NL', 'BE']
    np.random.seed(42)  # For reproducible results
    flows = np.random.randint(100, 1000, size=(len(timeindex), len(zones) * len(zones)))

    cols = pd.MultiIndex.from_product([zones, zones], names=['from_zone', 'to_zone'])
    flows_df = pd.DataFrame(flows, index=timeindex, columns=cols)

    print("Original flows DataFrame (first 3 columns):")
    print(flows_df.iloc[:, :3])

    # Filter to only flows FROM zones with renewable energy
    renewable_flows = filter_by_model_query(
        flows_df, model_df, 'has_renewable == True', match_on_level='from_zone'
    )
    print(f"\nFiltered to flows FROM renewable zones (shape: {renewable_flows.shape}):")
    print(renewable_flows.iloc[:, :3])
    print()

    # Example 2: Filter simple Series
    print("=== Example 2: Simple Series ===")
    prices_series = pd.Series([50, 45, 55, 60], index=pd.Index(['DE', 'FR', 'NL', 'BE'], name='zone'))
    print("Original prices Series:")
    print(prices_series)

    expensive_zones = filter_by_model_query(prices_series, model_df, 'price > 50')
    print("\nFiltered to expensive zones (price > 50):")
    print(expensive_zones)
    print()

    # Example 3: Filter Series with MultiIndex
    print("=== Example 3: Series with MultiIndex ===")
    volumes_series = pd.Series([1000, 2000, 1500, 3000], index=pd.MultiIndex.from_tuples(
        [('DE', 'peak'), ('FR', 'base'), ('NL', 'peak'), ('BE', 'base')],
        names=['zone', 'load_type']
    ))
    print("Original volumes Series:")
    print(volumes_series)

    non_renewable_volumes = filter_by_model_query(volumes_series, model_df, 'has_renewable == False')
    print("\nFiltered to non-renewable zones:")
    print(non_renewable_volumes)
