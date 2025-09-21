import numpy as np
import pandas as pd

from mescal.energy_data_handling.area_accounting.area_variable_base import AreaVariableCalculatorBase


class AreaPriceCalculator(AreaVariableCalculatorBase):
    """Calculates area-level prices from node prices using simple or weighted averaging.
    
    This calculator aggregates node-level electricity prices to area-level (e.g., bidding zones,
    countries) using either simple averaging or weighted averaging based on demand, supply, or
    other energy quantities. It's particularly useful in energy market analysis where different
    regions may have multiple price nodes that need to be consolidated into representative area
    prices.
    
    The class inherits from AreaVariableCalculatorBase and provides energy-aware price aggregation
    that handles edge cases like zero weights and missing data appropriately.
    
    Typical use cases:
    - Aggregating nodal prices to bidding zone prices
    - Creating country-level price indices from multiple market nodes
    - Volume-weighted price calculations for regional analysis
    
    Args:
        node_model_df: DataFrame with node-area mappings
        area_column: Column name containing area identifiers
        
    Example:

        >>> import pandas as pd
        >>> import numpy as np
        >>>
        >>> # Node model with area mapping
        >>> node_model = pd.DataFrame({
        ...     'bidding_zone': ['DE_LU', 'DE_LU', 'FR', 'FR']
        ... }, index=['DE1', 'DE2', 'FR1', 'FR2'])
        >>>
        >>> # Price calculator
        >>> calc = AreaPriceCalculator(node_model, 'bidding_zone')
        >>>
        >>> # Node prices
        >>> prices = pd.DataFrame({
        ...     'DE1': [50.0, 45.0], 'DE2': [52.0, 47.0],
        ...     'FR1': [55.0, 48.0], 'FR2': [53.0, 46.0]
        ... }, index=pd.date_range('2024-01-01', periods=2, freq='h'))
        >>>
        >>> # Simple average
        >>> area_prices = calc.calculate(prices)
        >>> print(area_prices)
                   bidding_zone  DE_LU FR
            datetime
            2024-01-01 00:00:00  51.0  54.0
            2024-01-01 01:00:00  46.0  47.0
    """
    
    def calculate(
        self,
        node_price_df: pd.DataFrame,
        weighting_factor_df: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """Calculate area prices with different weighting options.

        Aggregates node-level prices to area-level using simple averaging (when no weights
        provided) or weighted averaging (when weights provided). The method handles missing
        nodes gracefully and ensures proper handling of zero weights and NaN values.

        In case you want to exclude certain nodes from the aggregation (e.g. because they
        are virtual or synthetic nodes), you can simply remove them from the node_price_df
        before passing it to this method.

        Args:
            node_price_df: Node-level price time series with datetime index and node columns.
                Values represent electricity prices in €/MWh or similar units.
            weighting_factor_df: Optional weighting factor DataFrame with same structure as
                node_price_df. Common weighting factors include:
                - node_demand_df: Demand-weighted prices
                - node_supply_df: Supply-weighted prices  
                - node_capacity_df: Capacity-weighted prices
                If None, simple arithmetic average is used.

        Returns:
            DataFrame with area-level prices. Index matches input time series, columns
            represent areas with prices in same units as input.

        Raises:
            ValueError: If node_price_df structure is invalid
            KeyError: If required nodes are missing from weighting_factor_df

        Example:

            >>> # Simple average
            >>> area_prices = calc.calculate(node_prices)
            >>> 
            >>> # Demand-weighted average  
            >>> weighted_prices = calc.calculate(node_prices, node_demand)
        """
        self._validate_node_data(node_price_df, 'node_price_df')
        
        area_prices = {}
        
        for area in self.areas:
            area_nodes = self.get_area_nodes(area)
            area_nodes = [n for n in area_nodes if n in node_price_df.columns]
            
            if not area_nodes:
                continue
            
            prices = node_price_df[area_nodes]
            
            if weighting_factor_df is None:
                area_prices[area] = self._calculate_simple_average(prices)
            else:
                self._validate_node_data(weighting_factor_df, 'weighting_factor_df')
                area_prices[area] = self._calculate_weighted_average(
                    prices, weighting_factor_df[area_nodes]
                )

        result = pd.DataFrame(area_prices)
        result.columns.name = self.area_column
        return result
    
    def _calculate_simple_average(self, prices: pd.DataFrame) -> pd.Series:
        return prices.mean(axis=1)
    
    def _calculate_weighted_average(
        self, 
        prices: pd.DataFrame, 
        weights: pd.DataFrame
    ) -> pd.Series:
        """Calculate weighted average of prices using provided weights.
        
        Computes volume-weighted or otherwise weighted prices while handling edge cases
        appropriately. When weights sum to zero, the method defaults to weight of 1 to
        avoid division errors. When all prices are NaN for a time period, the result
        is also NaN.
        
        Args:
            prices: DataFrame with price time series for nodes in an area
            weights: DataFrame with weighting factors (e.g., demand, supply) with same
                structure as prices. Must have non-negative values.
                
        Returns:
            Series with weighted average prices over time
            
        Note:
            This method assumes weights are extensive quantities (like energy volumes)
            while prices are intensive quantities (like €/MWh).
        """
        weighted_sum = (prices * weights).sum(axis=1)
        weight_sum = weights.sum(axis=1).replace(0, 1)
        weighted_price = weighted_sum / weight_sum
        weighted_price[prices.isna().all(axis=1)] = np.nan
        return weighted_price


if __name__ == '__main__':
    print("=== MESCAL Area Price Calculator Examples ===\n")
    
    # Create sample node model with area mapping
    node_model_df = pd.DataFrame({
        'bidding_zone': ['DE_LU', 'DE_LU', 'FR', 'FR', 'BE', 'NL'],
        'country': ['DE', 'DE', 'FR', 'FR', 'BE', 'NL']
    }, index=['DE1', 'DE2', 'FR1', 'FR2', 'BE1', 'NL1'])
    
    print("Node model:")
    print(node_model_df)
    print()
    
    # Create sample price data (€/MWh)
    time_index = pd.date_range('2024-01-01', periods=6, freq='h')
    node_prices = pd.DataFrame({
        'DE1': [45.0, 50.0, 55.0, 48.0, 42.0, 46.0],
        'DE2': [47.0, 52.0, 57.0, 50.0, 44.0, 48.0],
        'FR1': [55.0, 58.0, 62.0, 56.0, 52.0, 54.0],
        'FR2': [53.0, 56.0, 60.0, 54.0, 50.0, 52.0],
        'BE1': [48.0, 53.0, 58.0, 51.0, 45.0, 49.0],
        'NL1': [50.0, 55.0, 60.0, 53.0, 47.0, 51.0]
    }, index=time_index)
    
    print("Node prices (€/MWh):")
    print(node_prices.round(2))
    print()
    
    # Example 1: Simple average aggregation to bidding zones
    print("1. Simple average aggregation to bidding zones:")
    price_calc_bz = AreaPriceCalculator(node_model_df, 'bidding_zone')
    bz_prices_simple = price_calc_bz.calculate(node_prices)
    print(bz_prices_simple.round(2))
    print()
    
    # Example 2: Weighted average with demand data
    print("2. Weighted average with synthetic demand data:")
    # Create synthetic demand data (MWh)
    np.random.seed(42)
    node_demand = pd.DataFrame({
        'DE1': np.random.uniform(800, 1200, 6),
        'DE2': np.random.uniform(700, 1100, 6),
        'FR1': np.random.uniform(900, 1300, 6),
        'FR2': np.random.uniform(600, 1000, 6),
        'BE1': np.random.uniform(400, 800, 6),
        'NL1': np.random.uniform(500, 900, 6)
    }, index=time_index)
    
    print("Node demand (MWh):")
    print(node_demand.round(1))
    print()
    
    bz_prices_weighted = price_calc_bz.calculate(node_prices, node_demand)
    print("Demand-weighted bidding zone prices (€/MWh):")
    print(bz_prices_weighted.round(2))
    print()
    
    # Example 3: Country-level aggregation
    print("3. Country-level aggregation:")
    price_calc_country = AreaPriceCalculator(node_model_df, 'country')
    country_prices = price_calc_country.calculate(node_prices, node_demand)
    print("Demand-weighted country prices (€/MWh):")
    print(country_prices.round(2))
    print()
    
    # Example 4: Handling missing nodes
    print("4. Handling missing nodes (excluding NL1):")
    partial_prices = node_prices.drop('NL1', axis=1)
    partial_demand = node_demand.drop('NL1', axis=1)
    
    partial_bz_prices = price_calc_bz.calculate(partial_prices, partial_demand)
    print("Bidding zone prices (NL excluded, shows only areas with data):")
    print(partial_bz_prices.round(2))
    print()
    
    # Example 5: Edge case - zero weights
    print("5. Edge case - handling zero demand periods:")
    zero_demand = node_demand.copy()
    zero_demand.iloc[2, :] = 0  # Set all demand to zero for one hour
    
    bz_prices_zero = price_calc_bz.calculate(node_prices, zero_demand)
    print("Prices with zero demand period (hour 2):")
    print(bz_prices_zero.round(2))
    print()
    print("Note: When weights sum to zero, method defaults to weight=1 to avoid division errors")
    
    # Example 6: Comparison of methods
    print("6. Comparison: Simple vs Weighted averaging for DE_LU:")
    comparison = pd.DataFrame({
        'Simple_Avg': bz_prices_simple['DE_LU'],
        'Demand_Weighted': bz_prices_weighted['DE_LU'],
        'Difference': bz_prices_weighted['DE_LU'] - bz_prices_simple['DE_LU']
    })
    print(comparison.round(3))
    print("Difference shows impact of demand weighting on area prices")
