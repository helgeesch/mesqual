import pandas as pd


def flatten_df(df: pd.DataFrame) -> pd.DataFrame:

    data = df.copy()
    if any(i is None for i in data.columns.names):
        data.columns.names = [f'column_level_{i}' if name is None else name for i, name in enumerate(df.columns.names)]
    if any(i is None for i in data.index.names):
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
