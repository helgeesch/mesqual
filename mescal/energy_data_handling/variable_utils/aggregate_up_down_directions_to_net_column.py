from typing import Literal
import pandas as pd

from mescal.utils.pandas_utils.set_new_column import set_column
from mescal.utils.multi_key_utils.common_base_key_finder import CommonBaseKeyFinder


class UpDownNetAppender:
    """Computes and appends net and total columns from bidirectional data.
    
    This utility class processes DataFrames containing bidirectional energy data
    (e.g., power flows, trade volumes) and creates derived columns representing
    net flows (up - down) and total flows (up + down). It automatically identifies
    column pairs based on configurable directional identifiers.
    
    Energy market context:
    Bidirectional energy data is common in electricity markets where flows,
    trades, or capacities can occur in both directions between nodes or areas.
    Net calculations show the overall direction and magnitude of transfers,
    while total calculations show overall activity levels.
    
    Common use cases:
    - Power flow analysis: Converting line flows to net flows
    - Trade balance: Net imports/exports from bilateral trade data  
    - Capacity utilization: Total utilization from bidirectional capacities
    - Market coupling: Net positions from bilateral exchanges
    
    The class uses CommonBaseKeyFinder to identify related column pairs and
    supports flexible naming conventions for input and output columns.
    
    Args:
        up_identifier: String identifier for "up" direction columns (default: '_up')
        down_identifier: String identifier for "down" direction columns (default: '_down')
        net_col_suffix: Suffix for generated net columns (default: '_net')
        net_col_prefix: Prefix for generated net columns (default: None)
        total_col_suffix: Suffix for generated total columns (default: '_total')
        total_col_prefix: Prefix for generated total columns (default: None)
        
    Raises:
        Exception: If neither net_col_suffix nor net_col_prefix is provided
        Exception: If neither total_col_suffix nor total_col_prefix is provided
        
    Example:
        >>> import pandas as pd
        >>> # Data with bidirectional flows
        >>> data = pd.DataFrame({
        ...     'DE_FR_up': [100, 200, 300],
        ...     'DE_FR_down': [30, 50, 100],
        ...     'FR_BE_up': [400, 500, 600],
        ...     'FR_BE_down': [150, 200, 300]
        ... })
        >>> 
        >>> appender = UpDownNetAppender()
        >>> result = appender.append_net_columns_from_up_down_columns(data)
        >>> print(result.columns)  # Includes DE_FR_net, FR_BE_net
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
            up_identifier: String identifier for "up" direction columns
            down_identifier: String identifier for "down" direction columns  
            net_col_suffix: Suffix for generated net columns
            net_col_prefix: Prefix for generated net columns
            total_col_suffix: Suffix for generated total columns
            total_col_prefix: Prefix for generated total columns
            
        Raises:
            Exception: If neither suffix nor prefix provided for net or total columns
        """
        self._up_identifier = up_identifier
        self._down_identifier = down_identifier

        self._net_col_suffix = net_col_suffix or ''
        self._net_col_prefix = net_col_prefix or ''
        if not any([self._net_col_suffix, self._net_col_prefix]):
            raise Exception("Either net_col_suffix or net_col_prefix must be provided")

        self._total_col_suffix = total_col_suffix or ''
        self._total_col_prefix = total_col_prefix or ''
        if not any([self._total_col_suffix, self._total_col_prefix]):
            raise Exception("Either total_col_suffix or total_col_prefix must be provided")

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
            
        Example:
            >>> data = pd.DataFrame({
            ...     'flow_up': [100, 200], 'flow_down': [30, 50]
            ... })
            >>> result = appender.append_net_columns_from_up_down_columns(data)
            >>> print(result['flow_net'])  # [70, 150]
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
            
        Example:
            >>> data = pd.DataFrame({
            ...     'flow_up': [100, 200], 'flow_down': [30, 50]  
            ... })
            >>> result = appender.append_total_columns_from_up_down_columns(data)
            >>> print(result['flow_total'])  # [130, 250]
        """
        return self._append_columns_from_up_down_columns(ts_df_with_up_down_columns, 'total')

    def _append_columns_from_up_down_columns(
            self,
            ts_df_with_up_down_columns: pd.DataFrame,
            which_agg: Literal['net', 'total']
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
                    ts_df_with_up_down_columns[down_col], fill_value=0,
                )
            elif which_agg == 'total':
                new_col = f'{self._total_col_prefix}{c}{self._total_col_suffix}'
                new_values = ts_df_with_up_down_columns[up_col].add(
                    ts_df_with_up_down_columns[down_col], fill_value=0,
                )
            else:
                raise NotImplementedError
            ts_df_with_up_down_columns = set_column(ts_df_with_up_down_columns, new_col, new_values)
        return ts_df_with_up_down_columns


if __name__ == '__main__':
    import numpy as np
    
    print("=== UpDownNetAppender Demo ===\n")
    
    # Example 1: Basic power flow data
    print("Example 1: Basic Power Flow Analysis")
    time_index = pd.date_range('2024-01-01', periods=5, freq='h')
    basic_data = pd.DataFrame({
        'DE_FR_up': [100, 200, 300, 150, 250],  # Power flow DE→FR
        'DE_FR_down': [30, 50, 100, 80, 40],   # Power flow FR→DE
        'FR_BE_up': [400, 500, 600, 300, 450], # Power flow FR→BE
        'FR_BE_down': [150, 200, 300, 100, 200] # Power flow BE→FR
    }, index=time_index)
    
    print("Original data:")
    print(basic_data)
    
    appender = UpDownNetAppender()
    result_basic = appender.append_net_columns_from_up_down_columns(basic_data)
    result_basic = appender.append_total_columns_from_up_down_columns(result_basic)
    
    print("\nWith net and total columns:")
    print(result_basic)
    
    # Example 2: Multi-level columns (common in energy data)
    print("\n" + "="*50)
    print("Example 2: Multi-level Columns (Country → Region)")
    
    # Create multi-level column structure
    countries = ['DE', 'FR']
    regions = ['North', 'South'] 
    directions = ['up', 'down']
    
    multi_columns = pd.MultiIndex.from_product(
        [countries, regions, directions],
        names=['country', 'region', 'direction']
    )
    
    multi_data = pd.DataFrame(
        np.random.uniform(50, 500, size=(5, len(multi_columns))),
        index=time_index,
        columns=multi_columns
    )
    
    print("Multi-level data structure:")
    print(multi_data.head())
    
    # Use custom identifiers for multi-level
    multi_appender = UpDownNetAppender(
        up_identifier='up',
        down_identifier='down', 
        net_col_suffix='net'
    )
    
    result_multi = multi_appender.append_net_columns_from_up_down_columns(multi_data)
    print(f"\nAfter adding net columns:")
    print(f"Column levels: {result_multi.columns.nlevels}")
    print(f"New columns include 'net': {any('net' in str(col) for col in result_multi.columns)}")
    print(result_multi.head())
    
    # Example 3: Real-world energy market scenario
    print("\n" + "="*50)
    print("Example 3: Energy Market Trade Balance")
    
    # Simulate hourly bilateral trade data
    hours = pd.date_range('2024-01-01', periods=24, freq='h')
    
    # Generate realistic trade patterns with daily cycles
    base_export = 200 + 100 * np.sin(np.arange(24) * 2 * np.pi / 24)  # Daily pattern
    base_import = 150 + 50 * np.sin((np.arange(24) + 12) * 2 * np.pi / 24)  # Opposite pattern
    
    trade_data = pd.DataFrame({
        'DE_export_to_FR_up': base_export + np.random.normal(0, 20, 24),
        'DE_import_from_FR_down': base_import + np.random.normal(0, 15, 24),
        'FR_export_to_BE_up': base_export * 0.7 + np.random.normal(0, 15, 24),
        'FR_import_from_BE_down': base_import * 0.8 + np.random.normal(0, 10, 24)
    }, index=hours)
    
    # Ensure non-negative flows
    trade_data = trade_data.clip(lower=0)
    
    print("Trade flow data (first 8 hours):")
    print(trade_data.head(8))
    
    # Create trade balance calculator with custom naming
    trade_appender = UpDownNetAppender(
        up_identifier='_up',
        down_identifier='_down',
        net_col_prefix='net_',
        total_col_prefix='total_'
    )
    
    trade_result = trade_appender.append_net_columns_from_up_down_columns(trade_data)
    trade_result = trade_appender.append_total_columns_from_up_down_columns(trade_result)
    
    print("\nTrade balance analysis (first 8 hours):")
    print(trade_result.head(8))
    
    # Show summary statistics
    print("\nDaily Trade Statistics:")
    net_cols = [col for col in trade_result.columns if 'net_' in col]
    total_cols = [col for col in trade_result.columns if 'total_' in col]
    
    for net_col, total_col in zip(net_cols, total_cols):
        net_avg = trade_result[net_col].mean()
        total_avg = trade_result[total_col].mean()
        net_direction = "net export" if net_avg > 0 else "net import"
        
        print(f"{net_col}: {net_avg:.1f} MW average ({net_direction})")
        print(f"{total_col}: {total_avg:.1f} MW average total activity")
        
    # Example 4: Error handling demonstration
    print("\n" + "="*50)
    print("Example 4: Edge Cases and Validation")
    
    # Test with missing pairs
    incomplete_data = pd.DataFrame({
        'complete_flow_up': [100, 200, 300],
        'complete_flow_down': [50, 75, 100],
        'orphan_up': [10, 20, 30],  # No matching _down column
        'standalone_data': [5, 10, 15]  # No directional identifier
    })
    
    result_incomplete = appender.append_net_columns_from_up_down_columns(incomplete_data)
    print("Data with incomplete pairs:")
    print("Original columns:", list(incomplete_data.columns))
    print("Result columns:", list(result_incomplete.columns))
    print("Net column created for 'complete_flow':", 'complete_flow_net' in result_incomplete.columns)
    
    # Show final comparison
    print("\n" + "="*50)
    print("Final Comparison: Net vs Total Analysis")
    sample_data = pd.DataFrame({
        'border_up': [100, 50, 200],
        'border_down': [80, 120, 150]
    })
    
    final_result = appender.append_net_columns_from_up_down_columns(sample_data)
    final_result = appender.append_total_columns_from_up_down_columns(final_result)
    
    print(final_result)
    print("\nInterpretation:")
    for i, row in final_result.iterrows():
        net = row['border_net']
        total = row['border_total'] 
        if net > 0:
            print(f"Hour {i}: Net export of {net:.0f} MW, total activity {total:.0f} MW")
        else:
            print(f"Hour {i}: Net import of {abs(net):.0f} MW, total activity {total:.0f} MW")
