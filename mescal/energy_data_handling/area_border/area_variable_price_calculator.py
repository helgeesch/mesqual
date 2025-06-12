from typing import Literal

import numpy as np
import pandas as pd

from mescal.energy_data_handling.area_border.area_variable_base import AreaVariableCalculatorBase


class AreaPriceCalculator(AreaVariableCalculatorBase):
    """Calculates area-level prices from node prices."""
    
    def calculate(
        self,
        node_price_df: pd.DataFrame,
        weighting_factor_df: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """Calculate area prices with different weighting options.
        
        Args:
            node_price_df: Node-level price time series
            weighting_factor_df: Optional weighting factor (e.g. node_demand_df, or node_supply_df)

        Returns:
            DataFrame with area prices
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
        """Calculate simple average of prices."""
        return prices.mean(axis=1)
    
    def _calculate_weighted_average(
        self, 
        prices: pd.DataFrame, 
        weights: pd.DataFrame
    ) -> pd.Series:
        """Calculate weighted average of prices."""
        weighted_sum = (prices * weights).sum(axis=1)
        weight_sum = weights.sum(axis=1).replace(0, 1)
        weighted_price = weighted_sum / weight_sum
        weighted_price[prices.isna().all(axis=1)] = np.nan
        return weighted_price
