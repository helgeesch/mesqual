from typing import Literal
import pandas as pd

from mescal.energy_data_handling.area_accounting.border_variable_base import AreaBorderVariableCalculatorBase


class BorderPriceSpreadCalculator(AreaBorderVariableCalculatorBase):
    """Calculates price spreads between areas for each border."""
    
    @property
    def variable_name(self) -> str:
        return "price_spread"
    
    def calculate(
        self,
        area_price_df: pd.DataFrame,
        spread_type: Literal['raw', 'absolute', 'directional_up', 'directional_down'] = 'raw'
    ) -> pd.DataFrame:
        """Calculate price spreads between areas.
        
        Args:
            area_price_df: Area-level prices
            spread_type: Type of spread calculation
                - raw: price_to - price_from (can be negative)
                - absolute: |price_to - price_from|
                - directional_up: max(price_to - price_from, 0)
                - directional_down: max(price_from - price_to, 0)

        Returns:
            DataFrame with price spreads for each border
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
        """Calculate all spread types at once.
        
        Returns:
            MultiIndex DataFrame with all spread types
        """
        results = {}
        for spread_type in ['raw', 'absolute', 'directional_up', 'directional_down']:
            results[spread_type] = self.calculate(area_price_df, spread_type)
        
        return pd.concat(results, axis=1, names=['spread_type'])
