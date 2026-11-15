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
        The frame in long format: one row per value, one column per index
            level and per column level, and the values in the column `value`.

    Examples:
        >>> columns = pd.MultiIndex.from_product([['C1', 'C2'], ['price']], names=['country', 'type'])
        >>> df = pd.DataFrame([[40.0, 50.0], [42.0, 55.0]], index=pd.Index([0, 1], name='hour'), columns=columns)
        >>> flatten_df(df)
           hour country   type  value
        0     0      C1  price   40.0
        1     1      C1  price   42.0
        2     0      C2  price   50.0
        3     1      C2  price   55.0
    """

    data = df.copy()
    if any(i is None for i in data.columns.names):
        if data.columns.nlevels == 1:
            data.columns.name = 'columns'
        else:
            data.columns.names = [
                f'column_level_{i}' if name is None else name for i, name in enumerate(df.columns.names)
            ]
    if any(i is None for i in data.index.names):
        if data.index.nlevels == 1:
            data.index.name = 'index'
        else:
            data.index.names = [f'index_level_{i}' if name is None else name for i, name in enumerate(df.index.names)]

    depth_cols = data.columns.nlevels
    idx_names = list(data.index.names)
    if depth_cols > 1:
        idx_cols = [(i,) + tuple('' for _ in range(depth_cols - 1)) for i in idx_names]
    else:
        idx_cols = idx_names
    data = data.reset_index().melt(id_vars=idx_cols)
    data = data.rename(columns={tup: name for tup, name in zip(idx_cols, idx_names)})

    return data
