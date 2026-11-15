import pandas as pd

from mesqual.energy_data_handling.area_accounting.area_variable_base import AreaVariableCalculatorBase


class AreaSumCalculator(AreaVariableCalculatorBase):
    """Area sums of node values, for extensive quantities such as demand or generation.

    To leave out a node, for example a virtual node, drop its column from the
    data.

    Args:
        node_model_df: DataFrame mapping nodes to areas
        area_column: Column name containing area identifiers

    Examples:
        >>> calculator = AreaSumCalculator(dataset.fetch('Node.Model'), 'country')
        >>> calculator.calculate(dataset.fetch('Node.Results.demand_hourly')).iloc[:2].round(2)
        country                         C1       C2
        time
        2024-03-28 00:00:00+01:00  1377.90  1583.60
        2024-03-28 01:00:00+01:00  1414.62  1628.73
    """

    def calculate(self, node_data_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate area sums from node-level extensive quantity data.

        Sums node-level values within each area to create area-level aggregates.
        This method is designed for extensive quantities where summation is the
        appropriate aggregation method (e.g., generation, demand, volumes).

        A node of the node model that is not in the data is left out. A column
        that is not a node of the node model is left out, and a warning is
        logged. An area without nodes in the data has no column. NaN values are left out of the
        sum. A time step where all nodes of an area are NaN gives NaN for that
        area. The result has the index of `node_data_df`, also when
        `node_data_df` has no columns.

        Args:
            node_data_df: DataFrame with node-level time series data. Index should
                be datetime, columns should be node identifiers. Values represent
                extensive quantities (MW, MWh, etc.) that should be summed.

        Returns:
            DataFrame with area-level aggregated data. Index matches input time series,
                columns represent areas. Units are preserved from input data.
        """

        self._validate_node_data(node_data_df, 'node_data_df')

        area_sums = {}
        for area in self.areas:
            area_nodes = self.get_area_nodes(area)
            area_nodes = [n for n in area_nodes if n in node_data_df.columns]
            if area_nodes:
                area_sums[area] = node_data_df[area_nodes].sum(axis=1, min_count=1)

        columns = pd.Index(list(area_sums), name=self.area_column)
        return pd.DataFrame(area_sums, index=node_data_df.index, columns=columns)
