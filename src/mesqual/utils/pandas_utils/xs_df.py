from typing import Union, Hashable, Literal
import pandas as pd

Axis = Union[int, Literal['index', 'columns', 'rows']]


def xs_df(
    df: pd.DataFrame,
    keys: Hashable | list[Hashable],
    axis: Axis = 0,
    level: Hashable = None,
) -> pd.DataFrame:
    """Extract cross-section from MultiIndex DataFrame with support for multiple keys.

    Works like `DataFrame.xs`, but also accepts a list of keys.

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
        >>> prices = study.scen.fetch('Node.Results.price')
        >>> xs_df(prices, 'base', axis='columns', level='dataset').columns.tolist()
        ['N1', 'N2', 'N3', 'N4']
        >>> xs_df(prices, ['base', 'short'], axis='columns', level='dataset').columns.get_level_values('dataset').unique().tolist()
        ['base', 'short']
    """
    if isinstance(keys, list):
        if axis in [0, 'index', 'rows']:
            return df.iloc[df.index.get_level_values(level).isin(keys)]
        return df.iloc[:, df.columns.get_level_values(level).isin(keys)]
    return df.xs(keys, level=level, axis=axis, drop_level=True)
