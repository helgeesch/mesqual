from typing import Iterable
from functools import reduce

import pandas as pd


def combine_dfs(dfs: Iterable[pd.Series | pd.DataFrame], keep_first: bool = True):
    size = sum(1 for _ in dfs)
    if size == 0:
        raise ValueError("You need to pass at least one DataFrame / Series.")
    if size == 1:
        return [df for df in dfs][0]

    def merge_func(df1, df2):
        if isinstance(df1, pd.DataFrame) and isinstance(df2, pd.DataFrame):
            column_intersection = set(df1.columns).intersection(df2.columns)
            index_intersection = set(df1.index).intersection(df2.index)
            if (len(column_intersection) == 0) and (len(index_intersection) > 0):
                return pd.concat([df1, df2], axis=1)
            elif (len(column_intersection) > 0) and (len(index_intersection) == 0):
                return pd.concat([df1, df2], axis=0)

        if keep_first:
            return df1.combine_first(df2)
        else:
            return df2.combine_first(df1)

    return reduce(merge_func, dfs)
