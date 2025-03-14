from typing import Union,Hashable, Literal
import pandas as pd

Axis = Union[int, Literal["index", "columns", "rows"]]


def xs_df(
        df: pd.DataFrame,
        keys: Hashable | list[Hashable],
        axis: Axis = 0,
        level: Hashable = None,
) -> pd.DataFrame:
    if isinstance(keys, list):
        if axis in [0, 'index', 'rows']:
            return df.iloc[df.index.get_level_values(level).isin(keys)]
        return df.iloc[:, df.columns.get_level_values(level).isin(keys)]
    return df.xs(keys, level=level, axis=axis, drop_level=True)
