"""Check whether a frame or series has numeric dtypes only."""

import pandas as pd


def pd_is_numeric(df: pd.DataFrame | pd.Series) -> bool:
    """`True` if every column has a numeric dtype that is not boolean.

    The check reads the dtypes, not the values. So a column of numbers with
    dtype `object` is not numeric.
    """
    dtypes = [df.dtype] if isinstance(df, pd.Series) else df.dtypes
    return all(pd.api.types.is_numeric_dtype(d) and not pd.api.types.is_bool_dtype(d) for d in dtypes)
