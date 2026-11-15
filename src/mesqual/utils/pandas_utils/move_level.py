"""Move the top level of an axis to the last position."""

import pandas as pd


def move_top_level_last(data: pd.Series | pd.DataFrame, axis: int = 0) -> pd.Series | pd.DataFrame:
    """Return `data` with the top level of an axis moved to the last position.

    The other levels keep their order. An axis with one level is unchanged.

    Args:
        data: A Series or DataFrame.
        axis: `0` for the rows, `1` for the columns.

    Examples:
        >>> prices = study.scen.fetch('Node.Results.price')
        >>> prices.columns.names
        FrozenList(['dataset', 'Node'])
        >>> move_top_level_last(prices, axis=1).columns.names
        FrozenList(['Node', 'dataset'])
    """
    ax = data.axes[axis]
    return data.set_axis(ax.reorder_levels(list(range(1, ax.nlevels)) + [0]), axis=axis)
