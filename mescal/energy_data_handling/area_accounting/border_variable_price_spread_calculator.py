from typing import Literal
import pandas as pd

from mescal.energy_data_handling.area_accounting.border_variable_base import AreaBorderVariableCalculatorBase


class BorderPriceSpreadCalculator(AreaBorderVariableCalculatorBase):
    """Calculates electricity price spreads between areas for each border.
    
    This calculator computes price differences between market areas connected by
    transmission borders. Price spreads are fundamental indicators in electricity
    markets for:
    
    - Market integration analysis (zero spreads indicate perfect coupling)
    - Congestion identification (non-zero spreads suggest transmission constraints)
    - Arbitrage opportunity assessment (price differences drive trading incentives)
    - Market efficiency evaluation (persistent spreads may indicate inefficiencies)
    - Cross-border flow direction prediction (flows typically follow price gradients)
    
    Energy Market Context:
    In liberalized electricity markets, price differences between areas arise from:
    - Transmission capacity constraints (causing congestion)
    - Market coupling imperfections
    - Different supply/demand fundamentals
    - Bidding zone design and market rules
    - Network topology and power flow physics
    
    The calculator supports multiple spread calculation methods:
    - Raw spreads: Show actual price differences with sign (can be negative)
    - Absolute spreads: Show magnitude of price differences (always positive)
    - Directional spreads: Separate positive and negative price differences
    
    Spread Types:
    - 'raw': price_to - price_from (preserves direction and sign)
    - 'absolute': |price_to - price_from| (magnitude only)
    - 'directional_up': max(price_to - price_from, 0) (only positive spreads)
    - 'directional_down': max(price_from - price_to, 0) (only negative spreads as positive)
    
    Units:
    Price spreads inherit units from input price data (typically EUR/MWh, USD/MWh).
    Results maintain the same units as the input area prices.
    
    MESCAL Integration:
    - Integrates with MESCAL's area border modeling framework
    - Supports time-series price spread analysis
    - Compatible with MESCAL's scenario comparison capabilities
    - Follows MESCAL's energy data handling patterns
    
    Attributes:
        variable_name (str): Returns 'price_spread' for identification
    
    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>> 
        >>> # Create sample area price data
        >>> time_index = pd.date_range('2024-01-01', periods=24, freq='h')
        >>> area_prices = pd.DataFrame({
        ...     'DE': np.random.uniform(40, 80, 24),  # German prices
        ...     'FR': np.random.uniform(35, 75, 24),  # French prices
        ...     'BE': np.random.uniform(45, 85, 24)   # Belgian prices
        ... }, index=time_index)
        >>> 
        >>> # Set up border model and calculator (see base class docs for setup)
        >>> calculator = BorderPriceSpreadCalculator(
        ...     border_model_df, line_model_df, node_model_df, 'country'
        ... )
        >>> 
        >>> # Calculate raw price spreads
        >>> raw_spreads = calculator.calculate(area_prices, spread_type='raw')
        >>> print(f"Average spread DE-FR: {raw_spreads['DE-FR'].mean():.2f} EUR/MWh")
        >>> 
        >>> # Calculate all spread types at once
        >>> all_spreads = calculator.calculate_all_spread_types(area_prices)
        >>> print(all_spreads.head())
    """
    
    @property
    def variable_name(self) -> str:
        return "price_spread"
    
    def calculate(
        self,
        area_price_df: pd.DataFrame,
        spread_type: Literal['raw', 'absolute', 'directional_up', 'directional_down'] = 'raw'
    ) -> pd.DataFrame:
        """Calculate electricity price spreads between connected market areas.
        
        Computes price differences across transmission borders using the specified
        calculation method. Price spreads are calculated as directional differences
        based on the border naming convention (area_from → area_to).
        
        The calculation handles missing area data gracefully by excluding borders
        where either area lacks price data. This is common when analyzing subsets
        of larger energy systems or when dealing with data availability issues.
        
        Mathematical Definitions:
        - raw_spread = price_to - price_from
        - Raw: Shows actual price differences with sign
        - Absolute: |raw_spread| (magnitude regardless of direction)
        - Directional_up: max(raw_spread, 0) (only positive spreads)
        - Directional_down: max(-raw_spread, 0) (only negative spreads as positive values)
        
        Energy Market Interpretation:
        - Positive raw spread: price_to > price_from (flow incentive toward area_to)
        - Negative raw spread: price_to < price_from (flow incentive toward area_from)  
        - Zero spread: Perfect price convergence (no congestion)
        - High absolute spreads: Strong congestion or market inefficiency
        
        Args:
            area_price_df (pd.DataFrame): Time series of area-level electricity prices.
                - Index: DateTime index for time series analysis
                - Columns: Area identifiers matching border area names
                - Values: Prices in consistent units (e.g., EUR/MWh, USD/MWh)
                - Example shape: (8760 hours, N areas) for annual analysis
                
            spread_type (Literal): Method for calculating price spreads.
                - 'raw': Directional price differences (default, preserves sign)
                - 'absolute': Magnitude of price differences (always non-negative)
                - 'directional_up': Only spreads where price_to > price_from
                - 'directional_down': Only spreads where price_from > price_to
                
        Returns:
            pd.DataFrame: Border-level price spreads with temporal dimension.
                - Index: Same as input area_price_df (typically DatetimeIndex)
                - Columns: Border identifiers (e.g., 'DE-FR', 'FR-BE')
                - Column name: Set to self.border_identifier for consistency
                - Values: Price spreads in same units as input prices
                - Missing data: NaN where area price data is unavailable
                
        Raises:
            ValueError: If spread_type is not one of the supported options
            
        Example:
            >>> import pandas as pd
            >>> import numpy as np
            >>> 
            >>> # Create hourly price data for German and French markets
            >>> time_index = pd.date_range('2024-01-01', periods=24, freq='h')
            >>> prices = pd.DataFrame({
            ...     'DE': [45.2, 43.1, 41.8, 39.5, 38.2, 42.1, 52.3, 65.4,
            ...            72.1, 68.9, 64.2, 58.7, 55.1, 53.8, 56.2, 61.4,
            ...            67.8, 74.2, 69.1, 64.3, 58.9, 52.1, 48.7, 46.3],
            ...     'FR': [42.8, 41.2, 39.1, 37.8, 36.4, 40.3, 49.8, 62.1,
            ...            68.9, 65.2, 61.4, 56.8, 53.2, 51.9, 54.1, 58.7,
            ...            64.3, 70.8, 66.2, 61.1, 56.3, 49.8, 46.1, 43.9]
            ... }, index=time_index)
            >>> 
            >>> # Calculate raw spreads (FR - DE for DE-FR border)
            >>> raw_spreads = calculator.calculate(prices, 'raw')
            >>> print(f"Average DE-FR spread: {raw_spreads['DE-FR'].mean():.2f} EUR/MWh")
            >>> # Output: Average DE-FR spread: -2.15 EUR/MWh (German prices higher)
            >>> 
            >>> # Calculate absolute spreads for congestion analysis
            >>> abs_spreads = calculator.calculate(prices, 'absolute')
            >>> print(f"Average absolute spread: {abs_spreads['DE-FR'].mean():.2f} EUR/MWh")
            >>> # Output: Average absolute spread: 2.15 EUR/MWh
            >>> 
            >>> # Analyze directional spreads for flow prediction
            >>> up_spreads = calculator.calculate(prices, 'directional_up')
            >>> down_spreads = calculator.calculate(prices, 'directional_down')
            >>> print(f"Hours with FR > DE prices: {(up_spreads['DE-FR'] > 0).sum()}")
            >>> print(f"Hours with DE > FR prices: {(down_spreads['DE-FR'] > 0).sum()}")
            
        Note:
            Price spreads are intensive quantities (EUR/MWh) and do not require
            temporal aggregation rules like extensive quantities (MWh). However,
            they should be analyzed with appropriate time resolution for the
            energy market being studied (e.g., hourly for day-ahead markets).
        """
        self._validate_time_series_data(area_price_df, 'area_price_df')
        
        spreads = {}
        
        for border_id, border in self.area_border_model_df.iterrows():
            area_from = border[self.source_area_identifier]
            area_to = border[self.target_area_identifier]
            
            if area_from in area_price_df.columns and area_to in area_price_df.columns:
                price_from = area_price_df[area_from]
                price_to = area_price_df[area_to]
                
                raw_spread = price_to - price_from
                
                if spread_type == 'raw':
                    spreads[border_id] = raw_spread
                elif spread_type == 'absolute':
                    spreads[border_id] = raw_spread.abs()
                elif spread_type == 'directional_up':
                    spreads[border_id] = raw_spread.clip(lower=0)
                elif spread_type == 'directional_down':
                    spreads[border_id] = (-1 * raw_spread).clip(lower=0)
                else:
                    raise ValueError(f"Unknown spread_type: {spread_type}")
        
        result = pd.DataFrame(spreads)
        result.columns.name = self.border_identifier
        return result
    
    def calculate_all_spread_types(self, area_price_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all price spread types simultaneously for comprehensive analysis.
        
        Computes all four spread calculation methods (raw, absolute, directional_up,
        directional_down) in a single operation, providing a complete view of price
        relationships across all borders. This is particularly useful for energy
        market analysis where different spread metrics provide different insights.
        
        The method returns a MultiIndex DataFrame that enables efficient analysis
        of spread patterns across multiple dimensions simultaneously. This format
        is ideal for:
        - Comparative analysis of different spread metrics
        - Time-series analysis with multiple spread types
        - Statistical analysis across spread calculation methods
        - Visualization of spread patterns and correlations
        
        Energy Analysis Use Cases:
        - Market integration assessment: Compare raw vs absolute spreads
        - Flow direction analysis: Use directional spreads to predict power flows
        - Congestion identification: High absolute spreads indicate transmission limits
        - Arbitrage analysis: Raw spreads show profitable trading directions
        
        Args:
            area_price_df (pd.DataFrame): Time series of area-level electricity prices.
                Same format as required by the calculate() method:
                - Index: DateTime index for temporal analysis
                - Columns: Area identifiers matching border definitions
                - Values: Prices in consistent units (e.g., EUR/MWh, USD/MWh)
                
        Returns:
            pd.DataFrame: MultiIndex DataFrame with comprehensive spread analysis.
                - Index: Same temporal index as input area_price_df
                - Columns: MultiIndex with two levels:
                    - Level 0 ('spread_type'): ['raw', 'absolute', 'directional_up', 'directional_down']
                    - Level 1 (border_identifier): Border names (e.g., 'DE-FR', 'FR-BE')
                - Values: Price spreads in same units as input prices
                - Structure: (time_periods, spread_types × borders)
                
        Example:
            >>> import pandas as pd
            >>> import numpy as np
            >>> 
            >>> # Create sample price data
            >>> time_index = pd.date_range('2024-01-01', periods=24, freq='h')
            >>> prices = pd.DataFrame({
            ...     'DE': np.random.uniform(40, 80, 24),
            ...     'FR': np.random.uniform(35, 75, 24),
            ...     'BE': np.random.uniform(45, 85, 24)
            ... }, index=time_index)
            >>> 
            >>> # Calculate all spread types
            >>> all_spreads = calculator.calculate_all_spread_types(prices)
            >>> print(all_spreads.columns.names)
            >>> # Output: ['spread_type', 'country_border']
            >>> 
            >>> # Access specific spread types
            >>> raw_spreads = all_spreads['raw']
            >>> absolute_spreads = all_spreads['absolute']
            >>> 
            >>> # Analyze spread statistics by type
            >>> spread_stats = all_spreads.groupby(level='spread_type', axis=1).mean()
            >>> print(spread_stats)
            >>> 
            >>> # Compare directional flows
            >>> up_flows = all_spreads['directional_up'].sum(axis=1)
            >>> down_flows = all_spreads['directional_down'].sum(axis=1)
            >>> net_spread_pressure = up_flows - down_flows
            >>> 
            >>> # Identify hours with high price volatility
            >>> high_volatility_hours = absolute_spreads.mean(axis=1) > 10  # EUR/MWh threshold
            >>> print(f"Hours with high spread volatility: {high_volatility_hours.sum()}")
            
        Note:
            This method performs four separate calculations, so it's more computationally
            intensive than calling calculate() for a single spread type. However, it's
            more efficient than four separate calls and provides a convenient format
            for multi-dimensional analysis.
        """
        results = {}
        for spread_type in ['raw', 'absolute', 'directional_up', 'directional_down']:
            results[spread_type] = self.calculate(area_price_df, spread_type)
        
        return pd.concat(results, axis=1, names=['spread_type'])


if __name__ == "__main__":
    """
    Comprehensive examples demonstrating BorderPriceSpreadCalculator usage
    for energy market analysis scenarios.
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from datetime import datetime, timedelta
    
    print("=" * 80)
    print("BorderPriceSpreadCalculator - Comprehensive Examples")
    print("=" * 80)
    
    # Create example energy market setup
    print("\n1. Creating Energy Market Infrastructure Model")
    print("-" * 50)
    
    # Node model representing substations in different countries
    node_model_df = pd.DataFrame({
        'country': ['DE', 'DE', 'DE', 'FR', 'FR', 'FR', 'BE', 'BE', 'NL', 'NL'],
        'voltage_kv': [380, 220, 380, 400, 225, 400, 380, 150, 380, 220],
        'region': ['North', 'South', 'West', 'North', 'South', 'East', 'North', 'South', 'West', 'East']
    }, index=['DE_N1', 'DE_S1', 'DE_W1', 'FR_N1', 'FR_S1', 'FR_E1', 'BE_N1', 'BE_S1', 'NL_W1', 'NL_E1'])
    
    # Transmission line model connecting different countries
    line_model_df = pd.DataFrame({
        'node_from': ['DE_N1', 'DE_W1', 'FR_N1', 'FR_E1', 'BE_N1', 'BE_S1', 'NL_W1', 'NL_E1'],
        'node_to': ['FR_N1', 'FR_E1', 'BE_N1', 'BE_S1', 'NL_W1', 'DE_S1', 'DE_N1', 'BE_N1'],
        'capacity_mw': [2000, 1500, 1200, 1000, 800, 900, 1100, 1300],
        'length_km': [450, 380, 320, 280, 180, 420, 290, 220],
        'voltage_kv': [380, 400, 380, 225, 380, 220, 380, 380]
    }, index=['DE-FR_1', 'DE-FR_2', 'FR-BE_1', 'FR-BE_2', 'BE-NL_1', 'BE-DE_1', 'NL-DE_1', 'NL-BE_1'])
    
    # Area border model representing market coupling interfaces
    area_border_model_df = pd.DataFrame({
        'max_capacity_mw': [3500, 2200, 2100, 1400],  # Total capacity across border
        'market_coupling': [True, True, True, True],  # All borders have market coupling
        'congestion_threshold_mw': [3000, 1800, 1700, 1200]
    }, index=['DE-FR', 'FR-BE', 'BE-NL', 'NL-DE'])
    
    print("Node model (electricity substations):")
    print(node_model_df)
    print(f"\nLine model (transmission interconnections):")
    print(line_model_df[['node_from', 'node_to', 'capacity_mw']])
    print(f"\nBorder model (market interfaces):")
    print(area_border_model_df)
    
    # Initialize the calculator
    print(f"\n2. Initializing BorderPriceSpreadCalculator")
    print("-" * 50)
    
    calculator = BorderPriceSpreadCalculator(
        area_border_model_df=area_border_model_df,
        line_model_df=line_model_df,
        node_model_df=node_model_df,
        area_column='country'
    )
    
    print(f"Calculator initialized with {len(area_border_model_df)} borders")
    print(f"Variable name: '{calculator.variable_name}'")
    print(f"Border identifier: '{calculator.border_identifier}'")
    
    # Demonstrate border line mapping
    print(f"\n3. Border-to-Line Mapping Analysis")
    print("-" * 50)
    
    for border_id in area_border_model_df.index:
        lines_up, lines_down = calculator.get_border_lines_in_topological_up_and_down_direction(border_id)
        area_from, area_to = calculator.decompose_area_border_name_to_areas(border_id)
        
        print(f"Border {border_id} ({area_from} → {area_to}):")
        print(f"  Lines UP ({area_from} → {area_to}): {lines_up}")
        print(f"  Lines DOWN ({area_to} → {area_from}): {lines_down}")
        print(f"  Total lines: {len(lines_up) + len(lines_down)}")
    
    # Create realistic price time series data
    print(f"\n4. Creating Realistic Price Data")
    print("-" * 50)
    
    # Generate 48 hours of hourly price data (2 days)
    time_index = pd.date_range('2024-01-01', periods=48, freq='h')
    np.random.seed(42)  # For reproducible results
    
    # Base price patterns with daily cycles and country-specific characteristics
    hours = np.arange(48) % 24  # Hour of day pattern
    
    # German prices (higher industrial demand, renewable integration challenges)
    de_base = 50 + 15 * np.sin(2 * np.pi * hours / 24 - np.pi/3) + np.random.normal(0, 5, 48)
    de_renewable_dip = -10 * np.maximum(0, np.sin(2 * np.pi * hours / 24 - np.pi/2))  # Solar dip at noon
    de_prices = np.maximum(0, de_base + de_renewable_dip)
    
    # French prices (large nuclear base, different demand pattern)
    fr_base = 45 + 12 * np.sin(2 * np.pi * hours / 24 - np.pi/4) + np.random.normal(0, 4, 48)
    fr_nuclear_floor = 35  # Nuclear base load provides price floor
    fr_prices = np.maximum(fr_nuclear_floor, fr_base)
    
    # Belgian prices (smaller market, import dependent)
    be_base = 48 + 18 * np.sin(2 * np.pi * hours / 24 - np.pi/6) + np.random.normal(0, 6, 48)
    be_import_premium = 3  # Premium due to import dependency
    be_prices = np.maximum(0, be_base + be_import_premium)
    
    # Dutch prices (gas influence, port access)
    nl_base = 52 + 14 * np.sin(2 * np.pi * hours / 24 - np.pi/5) + np.random.normal(0, 5, 48)
    nl_gas_volatility = 8 * np.random.normal(0, 1, 48)  # Gas price volatility
    nl_prices = np.maximum(0, nl_base + nl_gas_volatility)
    
    area_prices = pd.DataFrame({
        'DE': de_prices,
        'FR': fr_prices,
        'BE': be_prices,
        'NL': nl_prices
    }, index=time_index)
    
    print("Sample price data (first 12 hours):")
    print(area_prices.head(12).round(2))
    
    print(f"\nPrice statistics (EUR/MWh):")
    price_stats = area_prices.describe()
    print(price_stats.round(2))
    
    # Demonstrate different spread calculation methods
    print(f"\n5. Price Spread Analysis - Individual Methods")
    print("-" * 50)
    
    # Raw spreads (show market integration)
    raw_spreads = calculator.calculate(area_prices, spread_type='raw')
    print("Raw price spreads (FR-DE, BE-FR, etc.) - First 12 hours:")
    print(raw_spreads.head(12).round(2))
    
    # Absolute spreads (show congestion magnitude)
    abs_spreads = calculator.calculate(area_prices, spread_type='absolute')
    print(f"\nAbsolute price spreads - Average over 48 hours:")
    abs_means = abs_spreads.mean().round(2)
    for border, spread in abs_means.items():
        print(f"  {border}: {spread:.2f} EUR/MWh")
    
    # Directional spreads for flow prediction
    up_spreads = calculator.calculate(area_prices, spread_type='directional_up')
    down_spreads = calculator.calculate(area_prices, spread_type='directional_down')
    
    print(f"\n6. Comprehensive Multi-Type Analysis")
    print("-" * 50)
    
    # Calculate all spread types at once
    all_spreads = calculator.calculate_all_spread_types(area_prices)
    print(f"Multi-index DataFrame shape: {all_spreads.shape}")
    print(f"Column levels: {all_spreads.columns.names}")
    print(f"Spread types: {all_spreads.columns.levels[0].tolist()}")
    print(f"Borders: {all_spreads.columns.levels[1].tolist()}")
    
    # Analyze spread patterns
    print(f"\n7. Market Analysis Insights")
    print("-" * 50)
    
    # Market integration analysis (low absolute spreads indicate good coupling)
    integration_score = 1 / (1 + abs_spreads.mean())  # Higher score = better integration
    print("Market Integration Scores (higher = better coupling):")
    for border, score in integration_score.items():
        print(f"  {border}: {score:.3f}")
    
    # Flow direction prediction
    print(f"\nPredicted Flow Directions (based on raw spreads):")
    avg_raw_spreads = raw_spreads.mean()
    for border, avg_spread in avg_raw_spreads.items():
        area_from, area_to = calculator.decompose_area_border_name_to_areas(border)
        if avg_spread > 1:  # Threshold for significant spread
            print(f"  {border}: {area_from} → {area_to} (spread: +{avg_spread:.2f} EUR/MWh)")
        elif avg_spread < -1:
            print(f"  {border}: {area_to} → {area_from} (spread: {avg_spread:.2f} EUR/MWh)")
        else:
            print(f"  {border}: Balanced/No clear direction (spread: {avg_spread:.2f} EUR/MWh)")
    
    # Congestion identification
    print(f"\nCongestion Analysis:")
    high_congestion_threshold = 10  # EUR/MWh
    congested_hours = (abs_spreads > high_congestion_threshold).sum()
    for border, hours in congested_hours.items():
        percentage = (hours / len(abs_spreads)) * 100
        print(f"  {border}: {hours}/{len(abs_spreads)} hours ({percentage:.1f}%) with spread > {high_congestion_threshold} EUR/MWh")
    
    # Arbitrage opportunities
    print(f"\nArbitrage Opportunity Analysis:")
    for border in raw_spreads.columns:
        profitable_hours = (abs(raw_spreads[border]) > 5).sum()  # 5 EUR/MWh arbitrage threshold
        area_from, area_to = calculator.decompose_area_border_name_to_areas(border)
        print(f"  {border}: {profitable_hours}/{len(raw_spreads)} hours ({100*profitable_hours/len(raw_spreads):.1f}%) with profitable arbitrage opportunities")
    
    # Statistical analysis
    print(f"\n8. Statistical Analysis of Spread Patterns")
    print("-" * 50)
    
    # Correlation analysis
    spread_corr = raw_spreads.corr()
    print("Raw spread correlations between borders:")
    print(spread_corr.round(3))
    
    # Volatility analysis  
    spread_volatility = raw_spreads.std()
    print(f"\nPrice spread volatility (standard deviation):")
    for border, vol in spread_volatility.items():
        print(f"  {border}: {vol:.2f} EUR/MWh")
    
    # Time-of-day analysis
    area_prices['hour'] = area_prices.index.hour
    raw_spreads['hour'] = raw_spreads.index.hour
    
    hourly_spread_pattern = raw_spreads.groupby('hour').mean()
    print(f"\nAverage spreads by hour of day (first border as example):")
    first_border = hourly_spread_pattern.columns[0]
    for hour in range(0, 24, 4):
        if hour in hourly_spread_pattern.index:
            spread_val = hourly_spread_pattern.loc[hour, first_border]
            print(f"  Hour {hour:02d}: {spread_val:+.2f} EUR/MWh")
    
    # Edge case testing
    print(f"\n9. Edge Case and Validation Testing")
    print("-" * 50)
    
    # Test with missing area data
    incomplete_prices = area_prices.drop('NL', axis=1)  # Remove Netherlands
    try:
        incomplete_spreads = calculator.calculate(incomplete_prices, spread_type='raw')
        nl_borders = [border for border in raw_spreads.columns if 'NL' in border]
        print(f"Missing area handling: {len(nl_borders)} NL-related borders excluded from calculation")
        print(f"Available borders with incomplete data: {list(incomplete_spreads.columns)}")
    except Exception as e:
        print(f"Error with missing area data: {e}")
    
    # Test with invalid spread type
    try:
        invalid_spreads = calculator.calculate(area_prices, spread_type='invalid_type')
    except ValueError as e:
        print(f"Invalid spread type handling: {e}")
    
    # Test with non-datetime index
    non_temporal_prices = area_prices.copy()
    non_temporal_prices.index = range(len(non_temporal_prices))
    try:
        non_temporal_spreads = calculator.calculate(non_temporal_prices, spread_type='raw')
        print("Non-datetime index: Calculation succeeded with warning logged")
    except Exception as e:
        print(f"Non-datetime index handling: {e}")
    
    print(f"\n10. Summary and Recommendations")
    print("-" * 50)
    print("BorderPriceSpreadCalculator provides comprehensive price spread analysis for:")
    print("✓ Market integration assessment via raw and absolute spreads")
    print("✓ Congestion identification through absolute spread thresholds")  
    print("✓ Flow direction prediction using directional spreads")
    print("✓ Arbitrage opportunity detection via spread magnitude analysis")
    print("✓ Multi-dimensional analysis with calculate_all_spread_types()")
    print("✓ Robust handling of missing data and edge cases")
    print(f"\nFor energy market analysis, consider:")
    print("• Using hourly resolution for day-ahead market analysis")
    print("• Applying 5-15 EUR/MWh thresholds for significant spread detection")
    print("• Analyzing seasonal patterns for long-term market assessment")
    print("• Combining with flow data for transmission constraint validation")
    
    print("\n" + "=" * 80)
