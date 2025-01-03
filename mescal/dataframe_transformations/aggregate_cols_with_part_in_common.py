import pandas as pd

from mescal.utils.pandas_utils.set_new_column import set_column


class AggregatedColumnAppender:
    """
    Adds an aggregated column to a pandas DataFrame by summing columns
    that share a common identifier in their names.
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
        cols = df.columns.get_level_values(0).unique()
        cols_with_common_part = [x for x in cols if self._in_common_part in x]

        if df.columns.nlevels == 1:
            dff = df[cols_with_common_part].sum(axis=1)
        else:
            dff = df[cols_with_common_part].T.groupby(level=list(range(1, df.columns.nlevels))).sum().T

        new_col_name = f'{self._agg_col_name_prefix}{self._in_common_part}{self._agg_col_name_suffix}'
        df = set_column(df, new_col_name, dff)
        return df


if __name__ == '__main__':
    data = pd.DataFrame({
        'volume_curve': [100, 200, 300],
        'volume_block': [50, 70, 100],
        'bing_bong': [30, 40, 50],
        'king_kong': [20, 25, 30],
    })

    appender = AggregatedColumnAppender(in_common_part='volume', agg_col_name_suffix='_total')
    result_df = appender.add_aggregated_column(data)

    print(result_df)
