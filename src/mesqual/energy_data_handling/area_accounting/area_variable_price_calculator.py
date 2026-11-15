import numpy as np
import pandas as pd

from mesqual.energy_data_handling.area_accounting.area_variable_base import AreaVariableCalculatorBase


class AreaPriceCalculator(AreaVariableCalculatorBase):
    """Area prices from node prices: the mean or the weighted mean per area.

    Weights are for example the node demand or the node generation. To leave
    out a node, for example a virtual node, drop its column from the prices.

    Args:
        node_model_df: DataFrame with node-area mappings
        area_column: Column name containing area identifiers

    Examples:
        >>> calculator = AreaPriceCalculator(dataset.fetch('Node.Model'), 'country')
        >>> prices = dataset.fetch('Node.Results.price')
        >>> calculator.calculate(prices).iloc[:2].round(2)
        country                       C1     C2
        time
        2024-03-28 00:00:00+01:00  43.28  41.71
        2024-03-28 01:00:00+01:00  43.40  42.23
        >>> demand = dataset.fetch('Node.Results.demand_hourly')
        >>> calculator.calculate(prices, weighting_factor_df=demand).iloc[:2].round(2)  # demand-weighted
        country                       C1     C2
        time
        2024-03-28 00:00:00+01:00  44.18  42.32
        2024-03-28 01:00:00+01:00  44.34  43.23
    """

    def calculate(
        self,
        node_price_df: pd.DataFrame,
        weighting_factor_df: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """Calculate area prices with different weighting options.

        Aggregates node-level prices to area-level using simple averaging (when no weights
        provided) or weighted averaging (when weights provided). A node of the node model
        that is not in `node_price_df` is left out. A column that is not a node of the
        node model is left out, and a warning is logged.

        Args:
            node_price_df: Node-level price time series with datetime index and node columns.
                Values represent electricity prices in €/MWh or similar units.
            weighting_factor_df: Optional weighting factor DataFrame with same structure as
                node_price_df, for example the node demand or supply.
                If None, simple arithmetic average is used. A time step where
                the weights of an area sum to zero gives NaN for that area.

        Returns:
            DataFrame with area-level prices. Index matches input time series, columns
                represent areas with prices in same units as input.

        Raises:
            KeyError: If a node of `node_price_df` is missing from weighting_factor_df
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
                area_prices[area] = self._calculate_weighted_average(prices, weighting_factor_df[area_nodes])

        result = pd.DataFrame(area_prices)
        result.columns.name = self.area_column
        return result

    def _calculate_simple_average(self, prices: pd.DataFrame) -> pd.Series:
        return prices.mean(axis=1)

    def _calculate_weighted_average(self, prices: pd.DataFrame, weights: pd.DataFrame) -> pd.Series:
        """Calculate weighted average of prices using provided weights.

        Computes volume-weighted or otherwise weighted prices. A time period where
        the weights sum to zero has no weighted price and gives NaN. When all prices
        are NaN for a time period, the result is also NaN.

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
        weight_sum = weights.sum(axis=1).replace(0, np.nan)
        weighted_price = weighted_sum / weight_sum
        weighted_price[prices.isna().all(axis=1)] = np.nan
        return weighted_price
