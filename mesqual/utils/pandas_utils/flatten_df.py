import pandas as pd


def flatten_df(df: pd.DataFrame) -> pd.DataFrame:
    """Transform a time-series DataFrame into a flat format with one value per row.

    Converts a DataFrame with multi-level columns (objects/variables/properties)
    and time-based indices into a long-format DataFrame where each row contains
    a single value with its corresponding metadata.

    Args:
        df (pd.DataFrame): Input DataFrame with potentially multi-level columns
            and indices. Typically represents time-series data with multiple
            variables, objects, or properties.

    Returns:
        pd.DataFrame: Flattened DataFrame in long format where:

            - Each row represents one data point
            - Original index levels become columns
            - Original column levels become the 'variable' column
            - Data values are in the 'value' column

    Examples:

        >>> import pandas as pd
        >>> import numpy as np
        >>>
        >>> # Create sample multi-level DataFrame
        >>> dt_idx = pd.date_range('2024-01-01', periods=3, freq='h', name='datetime')
        >>> cols = pd.MultiIndex.from_product([['DE', 'FR'], ['price']], names=['zone', 'type'])
        >>> df = pd.DataFrame(np.random.rand(3, 2), index=dt_idx, columns=cols)
        >>> print(df.head())
            zone                    DE            FR
            type                 price volume  price volume
            datetime
            2024-01-01 00:00:00  37.45  95.07  73.20  59.87
            2024-01-01 06:00:00  15.60  15.60   5.81  86.62
            2024-01-01 12:00:00  60.11  70.81   2.06  96.99
            2024-01-01 18:00:00  83.24  21.23  18.18  18.34
        >>>
        >>> # Flatten the DataFrame
        >>> flat_df = flatten_df(df)
        >>> print(flat_df.head())
                          datetime zone    type  value
            0  2024-01-01 00:00:00   DE   price  37.45
            1  2024-01-01 06:00:00   DE   price  15.60
            2  2024-01-01 12:00:00   DE   price  60.11
            3  2024-01-01 18:00:00   DE   price  83.24
            4  2024-01-01 00:00:00   DE  volume  95.07
            5  2024-01-01 06:00:00   DE  volume  15.60
            6  2024-01-01 12:00:00   DE  volume  70.81
            7  2024-01-01 18:00:00   DE  volume  21.23
            8  2024-01-01 00:00:00   FR   price  73.20
            9  2024-01-01 06:00:00   FR   price   5.81
            10 2024-01-01 12:00:00   FR   price   2.06
            11 2024-01-01 18:00:00   FR   price  18.18
            12 2024-01-01 00:00:00   FR  volume  59.87
            13 2024-01-01 06:00:00   FR  volume  86.62
    """

    data = df.copy()
    if any(i is None for i in data.columns.names):
        if data.columns.nlevels == 1:
            data.columns.name = 'columns'
        else:
            data.columns.names = [f'column_level_{i}' if name is None else name for i, name in enumerate(df.columns.names)]
    if any(i is None for i in data.index.names):
        if data.index.nlevels == 1:
            data.index.name = 'index'
        else:
            data.index.names = [f'index_level_{i}' if name is None else name for i, name in enumerate(df.index.names)]

    depth_cols = data.columns.nlevels
    idx_names = list(data.index.names)
    if depth_cols > 1:
        idx_cols = [(i, ) + tuple('' for _ in range(depth_cols - 1)) for i in idx_names]
    else:
        idx_cols = idx_names
    data = data.reset_index().melt(id_vars=idx_cols)
    data = data.rename(columns={tup: name for tup, name in zip(idx_cols, idx_names)})

    return data


if __name__ == '__main__':
    import numpy as np

    print("Example: Flattening multi-level time-series DataFrame")
    print("=" * 55)

    # Create sample data with multi-level columns and datetime index
    dt_idx = pd.date_range('2024-01-01', periods=4, freq='6h', name='datetime')
    cols = pd.MultiIndex.from_product(
        [['DE', 'FR'], ['price', 'volume']],
        names=['zone', 'type']
    )

    # Generate sample data
    np.random.seed(42)  # For reproducible output
    df = pd.DataFrame(np.random.rand(len(dt_idx), 4) * 100,
                     index=dt_idx, columns=cols)

    print("\nOriginal DataFrame (wide format):")
    print(df.round(2))

    # Flatten the DataFrame
    flat_df = flatten_df(df)

    print(f"\nFlattened DataFrame (long format) - {len(flat_df)} rows:")
    print(flat_df.round(2))
