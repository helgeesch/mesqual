import pandas as pd

from mescal.utils.pandas_utils.set_new_column import set_column
from mescal.utils.multi_key_utils.common_base_key_finder import CommonBaseKeyFinder


class UpDownNetAppender:
    """
    Computes and appends net columns to a pandas DataFrame by subtracting
    corresponding "down" columns from "up" columns, based on specified identifiers.
    """

    def __init__(
            self,
            up_identifier: str = '_up',
            down_identifier: str = '_down',
            net_col_suffix: str = '_net',
            net_col_prefix: str = None,
    ):
        self._up_identifier = up_identifier
        self._down_identifier = down_identifier
        self._net_col_suffix = net_col_suffix if net_col_suffix else ''
        self._net_col_prefix = net_col_prefix if net_col_prefix else ''
        self._common_base_key_finder = CommonBaseKeyFinder(up_identifier, down_identifier)

    def append_net_columns_from_up_down_columns(
            self,
            ts_df_with_up_down_columns: pd.DataFrame,
    ) -> pd.DataFrame:

        up_id = self._up_identifier
        down_id = self._down_identifier

        _col_names = ts_df_with_up_down_columns.columns.get_level_values(0).unique()
        up_down_columns = self._common_base_key_finder.get_keys_for_which_all_association_tags_appear(_col_names)

        for c in up_down_columns:
            up_col = f'{c}{up_id}'
            down_col = f'{c}{down_id}'
            net_col = f'{self._net_col_prefix}{c}{self._net_col_suffix}'
            net_values = ts_df_with_up_down_columns[up_col].subtract(
                ts_df_with_up_down_columns[down_col], fill_value=0,
            )
            ts_df_with_up_down_columns = set_column(ts_df_with_up_down_columns, net_col, net_values)
        return ts_df_with_up_down_columns


if __name__ == '__main__':
    data = {
        'flow_up': [100, 200, 300],
        'flow_down': [30, 50, 100],
        'bingbong_up': [400, 500, 600],
        'bingbong_down': [150, 200, 300],
    }
    df = pd.DataFrame(data)
    appender = UpDownNetAppender()

    result_df = appender.append_net_columns_from_up_down_columns(df)

    print(result_df)
