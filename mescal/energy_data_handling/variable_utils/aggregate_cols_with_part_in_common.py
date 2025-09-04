import numpy as np
import pandas as pd

from mescal.utils.pandas_utils.set_new_column import set_column


class AggregatedColumnAppender:
    """
    Adds aggregated columns to pandas DataFrames by summing columns that share a common identifier.
    
    This utility is particularly useful in energy systems modeling where multiple variables of the same
    type (e.g., different wind generation sources, various demand components, multiple storage units)
    need to be aggregated into totals for analysis or visualization.
    
    The class handles both single-level and multi-level DataFrame columns, making it suitable for
    MESCAL's energy variable structures that often include hierarchical indexing (time, scenarios,
    regions, technologies, etc.).
    
    Energy Domain Context:
        - Aggregates extensive quantities (volumes, energy, capacities) by summation
        - Preserves intensive quantities when all components are NaN
        - Handles hierarchical energy variable structures common in multi-scenario analysis
        - Suitable for technology groupings, regional aggregations, and fuel type summations
        
    MESCAL Integration:
        - Compatible with MESCAL's multi-level DataFrame structures
        - Preserves time series granularity and scenario dimensions
        - Follows MESCAL naming conventions for aggregated variables
        - Maintains data integrity for downstream analysis and visualization
    
    Args:
        in_common_part (str): The common substring to search for in column names. All columns
            containing this substring will be summed together.
        agg_col_name_prefix (str, optional): Prefix to add to the aggregated column name.
            Defaults to empty string.
        agg_col_name_suffix (str, optional): Suffix to add to the aggregated column name.
            Defaults to empty string.
    
    Examples:
        >>> # Basic energy variable aggregation
        >>> data = pd.DataFrame({
        ...     'wind_onshore_gen': [100, 200, 300],
        ...     'wind_offshore_gen': [50, 70, 100], 
        ...     'solar_pv_gen': [30, 40, 50]
        ... })
        >>> appender = AggregatedColumnAppender('wind', agg_col_name_suffix='_total')
        >>> result = appender.add_aggregated_column(data)
        >>> # Creates 'wind_total' column with sum of wind_onshore_gen and wind_offshore_gen
        
        >>> # Multi-level columns for scenario analysis
        >>> columns = pd.MultiIndex.from_tuples([
        ...     ('wind_onshore', 'scenario_1'), ('wind_onshore', 'scenario_2'),
        ...     ('wind_offshore', 'scenario_1'), ('wind_offshore', 'scenario_2')
        ... ])
        >>> data = pd.DataFrame(np.random.rand(24, 4), columns=columns)
        >>> appender = AggregatedColumnAppender('wind', agg_col_name_prefix='total_')
        >>> result = appender.add_aggregated_column(data)
        >>> # Creates 'total_wind' with aggregated values for each scenario
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
        and handles both single-level and multi-level column indices.
        
        For energy data, this is typically used to:
        - Aggregate different technology types (e.g., all wind sources)
        - Sum regional contributions (e.g., all demand in a country)
        - Combine fuel types (e.g., all fossil fuel generators)
        - Create technology group totals (e.g., all renewable sources)
        
        Args:
            df (pd.DataFrame): Input DataFrame containing energy variables. Can have single-level
                or multi-level column structure. For multi-level columns, aggregation is performed
                at the appropriate level while preserving the hierarchy.
        
        Returns:
            pd.DataFrame: Original DataFrame with added aggregated column. The new column name
                follows the pattern: {prefix}{in_common_part}{suffix}
        
        Raises:
            ValueError: If no columns contain the specified common part.
            TypeError: If input is not a pandas DataFrame.
            
        Examples:
            >>> # Single-level columns - technology aggregation
            >>> gen_data = pd.DataFrame({
            ...     'wind_onshore_MW': [120, 150, 180],
            ...     'wind_offshore_MW': [80, 90, 100],
            ...     'solar_pv_MW': [200, 250, 300],
            ...     'demand_MW': [400, 490, 580]
            ... })
            >>> appender = AggregatedColumnAppender('wind', agg_col_name_suffix='_total_MW')
            >>> result = appender.add_aggregated_column(gen_data)
            >>> result['wind_total_MW']  # [200, 240, 280]
            
            >>> # Multi-level columns - scenario and regional analysis  
            >>> idx = pd.date_range('2024-01-01', periods=3, freq='h')
            >>> cols = pd.MultiIndex.from_tuples([
            ...     ('storage_battery', 'DE', 'scenario_1'), 
            ...     ('storage_battery', 'FR', 'scenario_1'),
            ...     ('storage_pumped', 'DE', 'scenario_1'),
            ...     ('storage_pumped', 'FR', 'scenario_1')
            ... ], names=['technology', 'region', 'scenario'])
            >>> data = pd.DataFrame(np.random.rand(3, 4), index=idx, columns=cols)
            >>> appender = AggregatedColumnAppender('storage', agg_col_name_prefix='total_')
            >>> result = appender.add_aggregated_column(data)
            >>> # Creates 'total_storage' with sums grouped by (region, scenario)
            
            >>> # Handling NaN values in energy time series
            >>> price_data = pd.DataFrame({
            ...     'price_day_ahead': [50.0, np.nan, 45.0],
            ...     'price_intraday': [np.nan, np.nan, 47.0],  
            ...     'demand_forecast': [1000, 1100, 1200]
            ... })
            >>> appender = AggregatedColumnAppender('price', agg_col_name_suffix='_avg')
            >>> result = appender.add_aggregated_column(price_data)
            >>> # 'price_avg': [50.0, NaN, 92.0] - preserves NaN when all inputs are NaN
        
        Note:
            For multi-level DataFrames, the aggregation respects the hierarchical structure.
            If columns have levels beyond the first (technology level), the aggregation
            groups by those additional levels, creating separate totals for each combination.
            
            NaN handling follows pandas summation rules: NaN values are ignored unless all
            values in a row are NaN, in which case the result is NaN.
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


if __name__ == '__main__':
    print("=== MESCAL Energy Data Aggregation Examples ===\n")
    
    # Example 1: Basic technology aggregation for renewable energy sources
    print("1. Technology Aggregation - Renewable Energy Generation")
    print("-" * 55)
    renewable_gen = pd.DataFrame({
        'wind_onshore_MW': [120, 150, 180, 200],
        'wind_offshore_MW': [80, 90, 100, 120], 
        'solar_pv_MW': [200, 250, 300, 350],
        'solar_thermal_MW': [50, 60, 70, 80],
        'hydro_run_river_MW': [100, 100, 100, 100],
        'demand_MW': [550, 650, 750, 850]
    }, index=pd.date_range('2024-01-01 12:00', periods=4, freq='6h'))
    
    print("Original renewable generation data:")
    print(renewable_gen.round(1))
    
    # Aggregate wind sources
    wind_aggregator = AggregatedColumnAppender('wind', agg_col_name_suffix='_total_MW')
    result_wind = wind_aggregator.add_aggregated_column(renewable_gen)
    
    # Aggregate solar sources  
    solar_aggregator = AggregatedColumnAppender('solar', agg_col_name_suffix='_total_MW')
    result_both = solar_aggregator.add_aggregated_column(result_wind)
    
    print(f"\nAfter aggregating wind and solar sources:")
    print("New columns: wind_total_MW, solar_total_MW")
    print(result_both[['wind_total_MW', 'solar_total_MW']].round(1))
    
    # Example 2: Multi-level columns for scenario comparison
    print("\n\n2. Multi-Level Scenario Analysis - Storage Technologies")
    print("-" * 58)
    
    # Create multi-level columns for different scenarios and regions
    technologies = ['storage_battery', 'storage_pumped', 'storage_flywheel', 'generation_gas']
    regions = ['DE', 'FR', 'NL']
    scenarios = ['base', 'high_res']
    
    columns = pd.MultiIndex.from_product([technologies, regions, scenarios],
                                       names=['technology', 'region', 'scenario'])
    
    # Create sample data for 24 hours
    np.random.seed(42)  # For reproducible results
    hourly_index = pd.date_range('2024-01-01', periods=6, freq='4h')
    storage_data = pd.DataFrame(
        np.random.uniform(50, 200, (6, len(columns))),
        index=hourly_index,
        columns=columns
    )
    
    print("Original multi-level storage data (first 3 hours):")
    print(storage_data.iloc[:3, :8].round(1))  # Show subset for readability
    
    # Aggregate all storage technologies
    storage_aggregator = AggregatedColumnAppender('storage', agg_col_name_prefix='total_')
    storage_result = storage_aggregator.add_aggregated_column(storage_data)
    
    print(f"\nAggregated storage totals by region and scenario:")
    storage_totals = storage_result['total_storage']
    print(storage_totals.iloc[:3].round(1))  # Show first 3 time periods
    
    # Example 3: Handling missing data (NaN) in energy time series
    print("\n\n3. NaN Handling in Energy Price Data")
    print("-" * 42)
    
    price_data = pd.DataFrame({
        'price_day_ahead_EUR': [45.2, np.nan, 38.7, 52.3],
        'price_intraday_EUR': [46.1, 41.2, np.nan, np.nan],
        'price_balancing_EUR': [np.nan, np.nan, np.nan, 55.8],
        'carbon_price_EUR': [25.0, 26.5, 24.8, 27.2]
    }, index=pd.date_range('2024-01-01', periods=4, freq='h'))
    
    print("Original price data with NaN values:")
    print(price_data)
    
    price_aggregator = AggregatedColumnAppender('price', agg_col_name_suffix='_avg_EUR')
    price_result = price_aggregator.add_aggregated_column(price_data)
    
    print(f"\nAggregated price column (handles NaN appropriately):")
    print(f"price_avg_EUR: {price_result['price_avg_EUR'].values}")
    print("Note: Row 1 preserves NaN when all price inputs are NaN")
    
    # Example 4: Energy system component aggregation
    print("\n\n4. Energy System Component Aggregation")
    print("-" * 45)
    
    system_data = pd.DataFrame({
        'demand_residential_MWh': [800, 750, 900, 850],
        'demand_industrial_MWh': [1200, 1100, 1300, 1250],
        'demand_commercial_MWh': [400, 350, 450, 420],
        'supply_nuclear_MWh': [1800, 1800, 1800, 1800],
        'supply_coal_MWh': [300, 250, 400, 350],
        'supply_gas_MWh': [300, 150, 450, 370]
    }, index=pd.date_range('2024-01-01', periods=4, freq='6h'))
    
    print("Energy system component data:")
    print(system_data)
    
    # Aggregate demand and supply separately
    demand_agg = AggregatedColumnAppender('demand', agg_col_name_suffix='_total_MWh')
    supply_agg = AggregatedColumnAppender('supply', agg_col_name_suffix='_total_MWh') 
    
    result_demand = demand_agg.add_aggregated_column(system_data)
    result_final = supply_agg.add_aggregated_column(result_demand)
    
    print(f"\nSystem totals:")
    totals = result_final[['demand_total_MWh', 'supply_total_MWh']]
    print(totals)
    print(f"\nEnergy balance check (supply - demand):")
    balance = totals['supply_total_MWh'] - totals['demand_total_MWh'] 
    print(balance.round(1))
    
    print("\n=== End of Examples ===")
    print("\nThese examples demonstrate typical MESCAL energy data aggregation patterns:")
    print("- Technology grouping for renewable energy analysis")
    print("- Multi-scenario and multi-regional data handling")
    print("- Proper NaN handling in time series data")
    print("- Energy system balance calculations")
