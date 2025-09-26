from typing import Literal
import pandas as pd

from mesqual.energy_data_handling.area_accounting.border_variable_base import AreaBorderVariableCalculatorBase


class BorderPriceSpreadCalculator(AreaBorderVariableCalculatorBase):
    """Calculates electricity price spreads between areas for each border.

    This calculator computes price differences between connected areas.
    Price spreads are fundamental indicators in electricity markets for:
    - Market integration analysis (zero spreads indicate perfect coupling)
    - Congestion identification (non-zero spreads suggest transmission constraints)
    - Arbitrage opportunity assessment (price differences drive trading incentives)
    - Market efficiency evaluation (persistent spreads may indicate inefficiencies)
    - Cross-border flow direction prediction (flows typically follow price gradients)

    The calculator supports multiple spread calculation methods (Spread Types):
    - 'raw': price_to - price_from (preserves direction and sign)
    - 'absolute': |price_to - price_from| (magnitude only)
    - 'directional_up': max(price_to - price_from, 0) (only positive spreads)
    - 'directional_down': max(price_from - price_to, 0) (only negative spreads as positive)

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

        Returns a MultiIndex DataFrame with all four spread calculation methods
        (raw, absolute, directional_up, directional_down) in a single DataFrame,
        providing a complete view of price relationships across all borders.

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
        """
        results = {}
        for spread_type in ['raw', 'absolute', 'directional_up', 'directional_down']:
            results[spread_type] = self.calculate(area_price_df, spread_type)

        return pd.concat(results, axis=1, names=['spread_type'])


if __name__ == "__main__":
    import pandas as pd

    from mesqual.energy_data_handling.area_accounting.border_model_generator import AreaBorderModelGenerator

    print("=" * 80)
    print("BorderPriceSpreadCalculator - Minimum Example")
    print("=" * 80)

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
    area_border_model_df = AreaBorderModelGenerator(
        node_model_df,
        line_model_df,
        area_column='country',
        node_from_col='node_from',
        node_to_col='node_to'
    ).generate_area_border_model()

    calculator = BorderPriceSpreadCalculator(
        area_border_model_df=area_border_model_df,
        line_model_df=line_model_df,
        node_model_df=node_model_df,
        area_column='country'
    )

    print(f"Calculator initialized with {len(area_border_model_df)} borders")
    print(f"Variable name: '{calculator.variable_name}'")
    print(f"Border identifier: '{calculator.border_identifier}'")

    # Create realistic price time series data
    print(f"\n4. Minimum example")
    print("-" * 50)

    area_prices = pd.DataFrame({
        'DE': [10, 20, -50, 30],
        'FR': [10, 10, 10, 10],
        'BE': [80, 80, 80, 80],
        'NL': [80, 80, 90, 30]
    })

    # Raw spreads (show market integration)
    raw_spreads = calculator.calculate(area_prices, spread_type='raw')
    print("Raw price spreads:")
    print(raw_spreads)

    all_spread_types = calculator.calculate_all_spread_types(area_prices)
    print("All price spreads:")
    print(all_spread_types)