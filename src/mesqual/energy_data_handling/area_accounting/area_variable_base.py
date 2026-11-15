import logging
from abc import ABC, abstractmethod
from typing import Any
import pandas as pd

from mesqual.energy_data_handling.area_accounting.area_membership import AreaMembership

logger = logging.getLogger(__name__)


class AreaVariableCalculatorBase(ABC, AreaMembership):
    """Base class of the calculators that aggregate node data to areas.

    It maps the nodes to their areas and checks the node data. A subclass
    implements `calculate`, for example a sum (`AreaSumCalculator`) or a
    mean (`AreaPriceCalculator`).

    Args:
        node_model_df: DataFrame containing node information with area assignments.
            Index should be node identifiers, must contain the specified area_column.
        area_column: Name of the column in node_model_df that contains area assignments.
            Each node has at most one area. A node without an area
            (`AreaMembership.has_area`) belongs to no area.

    Attributes:
        node_model_df: The input node model DataFrame
        area_column: Name of the area assignment column
        areas: Sorted list of the unique areas of the nodes with an area

    Raises:
        ValueError: If area_column is not found in node_model_df

    Examples:
        >>> from mesqual.energy_data_handling.area_accounting import AreaSumCalculator
        >>> calculator = AreaSumCalculator(dataset.fetch('Node.Model'), 'country')
        >>> calculator.areas
        ['C1', 'C2']
        >>> calculator.get_area_nodes('C1')
        ['N1', 'N3']
    """

    def __init__(self, node_model_df: pd.DataFrame, area_column: str):
        """Initialize the area variable calculator.

        Args:
            node_model_df: DataFrame with node-to-area mapping
            area_column: Column name containing area assignments

        Raises:
            ValueError: If area_column not found in node_model_df
        """
        self.node_model_df = node_model_df
        self.area_column = area_column
        self._node_areas = self.get_node_areas()
        self.areas = sorted(self._node_areas.unique())

    def get_area_nodes(self, area: str) -> list[str]:
        """Get all nodes belonging to a specific area.

        Args:
            area: Area name to get nodes for

        Returns:
            List of node IDs that belong to the specified area. Empty for a
                value that is no area (`AreaMembership.has_area`).
        """
        return self._node_areas.index[self._node_areas == area].tolist()

    @abstractmethod
    def calculate(self, **kwargs: Any) -> pd.DataFrame:
        """Calculate the area variable. Must be implemented by subclasses.

        This method should contain the specific logic for aggregating node-level
        data to area level for the particular variable type. The implementation
        will vary depending on whether the variable is extensive (additive like
        energy volumes) or intensive (averaged like prices).

        Args:
            **kwargs: Variable-specific parameters for the calculation

        Returns:
            DataFrame with area-level aggregated data. Index should be datetime
                for time series data, columns should be area identifiers.
        """
        pass

    def _validate_node_data(self, node_df: pd.DataFrame, data_name: str):
        """Validate that required nodes are present in node_model_df.

        Logs warnings for any nodes found in the data that are not in the node model.
        This is important for detecting data inconsistencies or model updates.

        Args:
            node_df: DataFrame containing node-level data to validate
            data_name: Descriptive name of the data being validated (for logging)
        """
        missing_nodes = set(node_df.columns) - set(self.node_model_df.index)
        if missing_nodes:
            logger.warning(f'{len(missing_nodes)} nodes missing in node_model_df from {data_name}')
