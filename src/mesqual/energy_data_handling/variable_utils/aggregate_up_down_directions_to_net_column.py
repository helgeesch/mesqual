from typing import Literal
import pandas as pd

from mesqual.utils.pandas_utils.set_new_column import set_column
from mesqual.utils.multi_key_utils.common_base_key_finder import CommonBaseKeyFinder


class UpDownNetAppender:
    """Appends net (up - down) and total (up + down) columns for pairs of up and down columns.

    A pair is two columns with the same base name and the up and down
    identifiers, for example the flow of a border in both directions
    (`CommonBaseKeyFinder`).

    Args:
        up_identifier: Suffix of the "up" direction columns (default: '_up')
        down_identifier: Suffix of the "down" direction columns (default: '_down')
        net_col_suffix: Suffix for generated net columns (default: '_net')
        net_col_prefix: Prefix for generated net columns (default: None)
        total_col_suffix: Suffix for generated total columns (default: '_total')
        total_col_prefix: Prefix for generated total columns (default: None)

    Raises:
        AttributeError: If neither suffix nor prefix provided for net or total columns

    Examples:
        >>> flows = pd.DataFrame({'C1 - C2_up': [100, 200], 'C1 - C2_down': [30, 50]})
        >>> UpDownNetAppender().append_net_columns_from_up_down_columns(flows)
           C1 - C2_up  C1 - C2_down  C1 - C2_net
        0         100            30           70
        1         200            50          150
    """

    def __init__(
        self,
        up_identifier: str = '_up',
        down_identifier: str = '_down',
        net_col_suffix: str = '_net',
        net_col_prefix: str = None,
        total_col_suffix: str = '_total',
        total_col_prefix: str = None,
    ):
        """Initialize the UpDownNetAppender with naming conventions.

        Args:
            up_identifier: Suffix of the "up" direction columns
            down_identifier: Suffix of the "down" direction columns
            net_col_suffix: Suffix for generated net columns
            net_col_prefix: Prefix for generated net columns
            total_col_suffix: Suffix for generated total columns
            total_col_prefix: Prefix for generated total columns

        Raises:
            AttributeError: If neither suffix nor prefix provided for net or total columns
        """
        self._up_identifier = up_identifier
        self._down_identifier = down_identifier

        self._net_col_suffix = net_col_suffix or ''
        self._net_col_prefix = net_col_prefix or ''
        if not any([self._net_col_suffix, self._net_col_prefix]):
            raise AttributeError('Either net_col_suffix or net_col_prefix must be provided')

        self._total_col_suffix = total_col_suffix or ''
        self._total_col_prefix = total_col_prefix or ''
        if not any([self._total_col_suffix, self._total_col_prefix]):
            raise AttributeError('Either total_col_suffix or total_col_prefix must be provided')

        self._common_base_key_finder = CommonBaseKeyFinder(up_identifier, down_identifier)

    def append_net_columns_from_up_down_columns(
        self,
        ts_df_with_up_down_columns: pd.DataFrame,
    ) -> pd.DataFrame:
        """Append net columns (up - down) to the DataFrame.

        Identifies all column pairs with up/down directional identifiers and
        creates corresponding net columns by subtracting down values from up values.
        Net columns represent the overall direction and magnitude of flows.

        Args:
            ts_df_with_up_down_columns: DataFrame containing bidirectional columns
                with up/down identifiers. Can have single or multi-level column index.

        Returns:
            DataFrame with original columns plus new net columns. Net values are
                positive when up > down, negative when down > up.
        """
        return self._append_columns_from_up_down_columns(ts_df_with_up_down_columns, 'net')

    def append_total_columns_from_up_down_columns(
        self,
        ts_df_with_up_down_columns: pd.DataFrame,
    ) -> pd.DataFrame:
        """Append total columns (up + down) to the DataFrame.

        Identifies all column pairs with up/down directional identifiers and
        creates corresponding total columns by adding up and down values.
        Total columns represent the overall activity level regardless of direction.

        Args:
            ts_df_with_up_down_columns: DataFrame containing bidirectional columns
                with up/down identifiers. Can have single or multi-level column index.

        Returns:
            DataFrame with original columns plus new total columns. Total values
                represent the sum of absolute flows in both directions.

        Examples:
            >>> flows = pd.DataFrame({'C1 - C2_up': [100, 200], 'C1 - C2_down': [30, 50]})
            >>> UpDownNetAppender().append_total_columns_from_up_down_columns(flows)['C1 - C2_total'].tolist()
            [130, 250]
        """
        return self._append_columns_from_up_down_columns(ts_df_with_up_down_columns, 'total')

    def _append_columns_from_up_down_columns(
        self, ts_df_with_up_down_columns: pd.DataFrame, which_agg: Literal['net', 'total']
    ) -> pd.DataFrame:
        """Internal method to append either net or total columns.

        Args:
            ts_df_with_up_down_columns: DataFrame with bidirectional data
            which_agg: Type of aggregation to perform ('net' or 'total')

        Returns:
            DataFrame with new aggregated columns appended

        Raises:
            NotImplementedError: If which_agg is not 'net' or 'total'
        """

        up_id = self._up_identifier
        down_id = self._down_identifier

        _col_names = ts_df_with_up_down_columns.columns.get_level_values(0).unique()
        up_down_columns = self._common_base_key_finder.get_keys_for_which_all_association_tags_appear(_col_names)

        for c in up_down_columns:
            up_col = f'{c}{up_id}'
            down_col = f'{c}{down_id}'
            if which_agg == 'net':
                new_col = f'{self._net_col_prefix}{c}{self._net_col_suffix}'
                new_values = ts_df_with_up_down_columns[up_col].subtract(
                    ts_df_with_up_down_columns[down_col],
                    fill_value=0,
                )
            elif which_agg == 'total':
                new_col = f'{self._total_col_prefix}{c}{self._total_col_suffix}'
                new_values = ts_df_with_up_down_columns[up_col].add(
                    ts_df_with_up_down_columns[down_col],
                    fill_value=0,
                )
            else:
                raise NotImplementedError
            ts_df_with_up_down_columns = set_column(ts_df_with_up_down_columns, new_col, new_values)
        return ts_df_with_up_down_columns
