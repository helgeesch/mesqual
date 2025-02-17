import pandas as pd


def flatten_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform you time-series dataframe with objects and/or variables and/or properties on columns (column-levels)
    and time-indices on index into a flat DataFrame with exactly one "value" per row.
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
    dt_idx = pd.date_range('2024-01-01', '2024-01-02', freq='h', name='datetime')
    cols = pd.MultiIndex.from_product(
        [['DE', 'FR'], ['price', 'volume']],
        names=['zone', 'type']
    )
    df = pd.DataFrame(np.random.rand(len(dt_idx), 4), index=dt_idx, columns=cols)
    flat_df = flatten_df(df)
    print(flat_df)
