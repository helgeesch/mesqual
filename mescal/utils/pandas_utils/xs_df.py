from typing import Union,Hashable, Literal
import pandas as pd

Axis = Union[int, Literal["index", "columns", "rows"]]


def xs_df(
        df: pd.DataFrame,
        keys: Hashable | list[Hashable],
        axis: Axis = 0,
        level: Hashable = None,
) -> pd.DataFrame:
    """Extract cross-section from MultiIndex DataFrame with support for multiple keys.

    This function provides a flexible interface to pandas .xs() method with enhanced
    functionality for MESCAL's MultiIndex data structures. It supports both single
    and multiple key selection, making it particularly useful for energy systems
    analysis where data often has complex hierarchical structures.

    Args:
        df: Input DataFrame with MultiIndex (either on index or columns).
        keys: Single key or list of keys to select from the specified level.
            For single keys, uses pandas .xs() method with drop_level=True.
            For multiple keys, uses .isin() for efficient selection.
        axis: Axis to operate on. Can be 0/'index'/'rows' for index operations
            or 1/'columns' for column operations. Defaults to 0.
        level: Name or position of the MultiIndex level to select from.
            Must be specified for MultiIndex operations.

    Returns:
        DataFrame with cross-section data. For single keys, the specified level
        is dropped. For multiple keys, the level is preserved.

    Examples:
        Single dataset selection from MESCAL multi-scenario data:
        >>> multi_scenario_prices = study.scen.fetch('buses_t.marginal_price')
        >>> base_prices = xs_df(multi_scenario_prices, 'base', level='dataset')

        Multiple scenario selection:
        >>> scenarios = ['base', 'high_renewable', 'low_cost']
        >>> selected_data = xs_df(multi_scenario_prices, scenarios, level='dataset')

        Column-wise selection for specific buses:
        >>> bus_names = ['Bus_1', 'Bus_2', 'Bus_3']
        >>> selected_buses = xs_df(price_data, bus_names, axis='columns', level='Bus')
    """
    if isinstance(keys, list):
        if axis in [0, 'index', 'rows']:
            return df.iloc[df.index.get_level_values(level).isin(keys)]
        return df.iloc[:, df.columns.get_level_values(level).isin(keys)]
    return df.xs(keys, level=level, axis=axis, drop_level=True)


if __name__ == "__main__":
    # Create sample MultiIndex DataFrame mimicking MESCAL energy data structure
    import numpy as np

    # Create MultiIndex with scenarios and buses
    scenarios = ['base', 'high_renewable', 'low_cost']
    buses = ['Bus_1', 'Bus_2', 'Bus_3']
    timestamps = pd.date_range('2023-01-01', periods=24, freq='H')

    # Create sample price data
    np.random.seed(42)
    index = pd.MultiIndex.from_product(
        [timestamps, scenarios],
        names=['snapshot', 'dataset']
    )
    columns = pd.Index(buses, name='Bus')

    price_data = pd.DataFrame(
        np.random.uniform(20, 80, size=(len(index), len(columns))),
        index=index,
        columns=columns
    )

    print("Original MultiIndex DataFrame shape:", price_data.shape)
    print("Index names:", price_data.index.names)
    print("Column name:", price_data.columns.name)
    print()

    # Example 1: Single dataset selection
    print("=== Single Dataset Selection ===")
    base_scenario = xs_df(price_data, 'base', level='dataset')
    print(f"Base scenario shape: {base_scenario.shape}")
    print(f"Index names after selection: {base_scenario.index.names}")
    print()

    # Example 2: Multiple dataset selection
    print("=== Multiple Dataset Selection ===")
    selected_scenarios = xs_df(price_data, ['base', 'high_renewable'], level='dataset')
    print(f"Selected scenarios shape: {selected_scenarios.shape}")
    print(f"Unique datasets: {selected_scenarios.index.get_level_values('dataset').unique().tolist()}")
    print()

    # Example 3: Column-wise selection for specific buses
    print("=== Column-wise Bus Selection ===")
    selected_buses = xs_df(price_data, ['Bus_1', 'Bus_3'], axis='columns', level='Bus')
    print(f"Selected buses shape: {selected_buses.shape}")
    print(f"Selected bus columns: {selected_buses.columns.tolist()}")
