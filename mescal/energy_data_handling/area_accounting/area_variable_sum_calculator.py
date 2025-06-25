import pandas as pd

from mescal.energy_data_handling.area_accounting.area_variable_base import AreaVariableCalculatorBase


class AreaSumCalculator(AreaVariableCalculatorBase):
    """General calculator for summing node-level values to area level.

    This calculator can be used for any variable that needs to be aggregated
    by summing across nodes within each area (e.g., volumes, surplus, generation, etc.).
    """

    def calculate(self, node_data_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate area sums from node-level data.

        Args:
            node_data_df: A single DataFrame to sum all nodes per area.

        Returns:
            DataFrame with summed values per area
        """

        self._validate_node_data(node_data_df, 'node_data_df')

        area_sums = {}
        for area in self.areas:
            area_nodes = self.get_area_nodes(area)
            area_nodes = [n for n in area_nodes if n in node_data_df.columns]
            if area_nodes:
                area_sums[area] = node_data_df[area_nodes].sum(axis=1)

        result = pd.DataFrame(area_sums)
        result.columns.name = self.area_column
        return result


if __name__ == '__main__':
    import numpy as np

    index = pd.date_range('2024-01-01', periods=24, freq='h')

    node_model = pd.DataFrame({
        'country': ['DE', 'DE', 'FR', 'FR', 'BE'],
        'region': ['North', 'South', 'North', 'South', 'Central']
    }, index=['DE1', 'DE2', 'FR1', 'FR2', 'BE1'])

    print("Example: Volume Calculator")
    print("-" * 40)

    node_supply = pd.DataFrame({
        node: np.random.uniform(100, 200, 24)
        for node in node_model.index
    }, index=index)

    area_supply = AreaSumCalculator(node_model, 'region', 'supply').calculate(node_supply)
    print("Area supply:")
    print(area_supply.head())
