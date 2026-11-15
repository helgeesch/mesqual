import numpy as np
import pandas as pd

from mesqual.utils.pandas_utils.set_new_column import set_column


class AggregatedColumnAppender:
    """Appends the sum of all columns whose name contains a common part.

    Use it for extensive quantities, for example the sum of all wind
    generation columns. A row where all summed values are NaN gives NaN.
    With multi-level columns, the sum is per combination of the other levels.

    Args:
        in_common_part (str): The common substring to search for in column names. All columns
            containing this substring will be summed together.
        agg_col_name_prefix (str, optional): Prefix to add to the aggregated column name.
            Defaults to empty string.
        agg_col_name_suffix (str, optional): Suffix to add to the aggregated column name.
            Defaults to empty string.

    Examples:
        >>> import numpy as np
        >>> generation = pd.DataFrame({'wind_onshore': [100.0, np.nan], 'wind_offshore': [50.0, np.nan], 'solar': [30.0, 40.0]})
        >>> AggregatedColumnAppender('wind', agg_col_name_suffix='_total').add_aggregated_column(generation)
           wind_onshore  wind_offshore  solar  wind_total
        0         100.0           50.0   30.0       150.0
        1           NaN            NaN   40.0         NaN
    """

    def __init__(
        self,
        in_common_part: str,
        agg_col_name_prefix: str = None,
        agg_col_name_suffix: str = None,
    ):
        self._in_common_part = in_common_part
        self._agg_col_name_prefix = agg_col_name_prefix or ''
        self._agg_col_name_suffix = agg_col_name_suffix or ''

    def add_aggregated_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add an aggregated column to the DataFrame by summing matching columns.

        This method identifies all columns containing the specified common part and sums them
        into a new aggregated column. The aggregation preserves the DataFrame's structure
        and handles both single-level and multi-level column indices. Without a
        matching column, single-level columns get a new column of NaN, and
        multi-level columns get no new column.

        Args:
            df (pd.DataFrame): Input DataFrame containing energy variables. Can have single-level
                or multi-level column structure. For multi-level columns, aggregation is performed
                at the appropriate level while preserving the hierarchy.

        Returns:
            pd.DataFrame: Original DataFrame with added aggregated column. The new column name
                follows the pattern: {prefix}{in_common_part}{suffix}

        Examples:
            >>> columns = pd.MultiIndex.from_tuples([('wind_onshore', 'C1'), ('wind_offshore', 'C1'), ('wind_onshore', 'C2')])
            >>> generation = pd.DataFrame([[1.0, 2.0, 3.0]], columns=columns)
            >>> AggregatedColumnAppender('wind', agg_col_name_prefix='total_').add_aggregated_column(generation)['total_wind']
                C1   C2
            0  3.0  3.0
        """
        cols = df.columns.get_level_values(0).unique()
        cols_with_common_part = [x for x in cols if self._in_common_part in x]

        df_in_common = df[cols_with_common_part]
        if df.columns.nlevels == 1:
            dff = df_in_common.sum(axis=1)
            dff.loc[df_in_common.isna().all(axis=1)] = np.nan
        else:
            _groupby = list(range(1, df.columns.nlevels))
            dff = df_in_common.T.groupby(level=_groupby).sum().T
            _all_na = df_in_common.isna().T.groupby(level=_groupby).all().T
            if _all_na.any().any():
                for c in _all_na.columns:
                    dff.loc[_all_na[c], c] = np.nan

        new_col_name = f'{self._agg_col_name_prefix}{self._in_common_part}{self._agg_col_name_suffix}'
        df = set_column(df, new_col_name, dff)
        return df
