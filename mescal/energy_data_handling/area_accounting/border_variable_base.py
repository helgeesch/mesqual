from abc import ABC, abstractmethod
from typing import Hashable

import pandas as pd

from mescal.energy_data_handling.area_accounting.border_model_generator import AreaBorderNamingConventions
from mescal.utils.logging import get_logger

logger = get_logger(__name__)


class AreaBorderVariableCalculatorBase(ABC, AreaBorderNamingConventions):
    """Abstract base class for calculating energy variables at area border level.
    
    This base class provides functionality for aggregating line-level energy data 
    (flows, capacities, price spreads) to area border level. An area border represents
    the interface between two areas (countries, bidding zones, etc.).
    
    The class handles the complex mapping from transmission lines to area borders,
    including proper handling of line directionality. Lines are classified as either
    "up" or "down" relative to the border direction based on their node endpoints.
    
    Border directionality:
    - "Up" direction: From area_from to area_to (as defined in border naming)
    - "Down" direction: From area_to to area_from
    - Line direction is determined by comparing line endpoints to border areas
    
    Args:
        area_border_model_df: DataFrame containing area border definitions.
            Index should be border identifiers (e.g., 'DE-FR', 'FR-BE').
        line_model_df: DataFrame containing transmission line information.
            Must include node_from_col and node_to_col columns.
        node_model_df: DataFrame containing node information with area assignments.
            Must include area_column for mapping nodes to areas.
        area_column: Column name in node_model_df containing area assignments.
        node_from_col: Column name in line_model_df for line starting node.
        node_to_col: Column name in line_model_df for line ending node.
        
    Attributes:
        area_border_model_df: Border model DataFrame
        line_model_df: Line model DataFrame  
        node_model_df: Node model DataFrame
        area_column: Name of area assignment column
        node_from_col: Name of line from-node column
        node_to_col: Name of line to-node column
        node_to_area_map: Dictionary mapping node IDs to area names
        
    Raises:
        ValueError: If required columns are missing from input DataFrames
        
    Example:
        >>> import pandas as pd
        >>> # Define borders between areas
        >>> border_model = pd.DataFrame(index=['DE-FR', 'FR-BE'])
        >>> 
        >>> # Define transmission lines  
        >>> line_model = pd.DataFrame({
        ...     'node_from': ['DE1', 'FR1'],
        ...     'node_to': ['FR1', 'BE1'],
        ...     'capacity': [1000, 800]
        ... }, index=['Line1', 'Line2'])
        >>> 
        >>> # Node-to-area mapping
        >>> node_model = pd.DataFrame({
        ...     'country': ['DE', 'FR', 'BE']
        ... }, index=['DE1', 'FR1', 'BE1'])
        >>> 
        >>> # Subclass for specific calculation
        >>> class MyBorderCalculator(AreaBorderVariableCalculatorBase):
        ...     @property
        ...     def variable_name(self):
        ...         return "my_variable"
        ...     def calculate(self, **kwargs):
        ...         return pd.DataFrame()
        >>> 
        >>> calculator = MyBorderCalculator(
        ...     border_model, line_model, node_model, 'country'
        ... )
    """

    def __init__(
        self,
        area_border_model_df: pd.DataFrame,
        line_model_df: pd.DataFrame,
        node_model_df: pd.DataFrame,
        area_column: str,
        node_from_col: str = 'node_from',
        node_to_col: str = 'node_to'
    ):
        """Initialize the area border variable calculator.
        
        Args:
            area_border_model_df: DataFrame with border definitions
            line_model_df: DataFrame with line information including endpoints
            node_model_df: DataFrame with node-to-area mapping
            area_column: Column name for area assignments in node_model_df
            node_from_col: Column name for line starting node in line_model_df
            node_to_col: Column name for line ending node in line_model_df
            
        Raises:
            ValueError: If required columns are missing from DataFrames
        """
        super().__init__(area_column)
        self.area_border_model_df = area_border_model_df
        self.line_model_df = line_model_df
        self.node_model_df = node_model_df
        self.area_column = area_column
        self.node_from_col = node_from_col
        self.node_to_col = node_to_col
        self.node_to_area_map = self._create_node_to_area_map()
        self._validate_inputs()

    def _validate_inputs(self):
        """Validate input parameters during initialization.
        
        Raises:
            ValueError: If required columns are missing from DataFrames
        """
        if self.area_column not in self.node_model_df.columns:
            raise ValueError(f"Column '{self.area_column}' not found in node_model_df")
        if self.node_from_col not in self.line_model_df.columns:
            raise ValueError(f"Column '{self.node_from_col}' not found in line_model_df")
        if self.node_to_col not in self.line_model_df.columns:
            raise ValueError(f"Column '{self.node_to_col}' not found in line_model_df")

    def _create_node_to_area_map(self) -> dict[Hashable, str]:
        """Create a mapping dictionary from node IDs to area names.
        
        Returns:
            Dictionary with node IDs as keys and area names as values.
            Nodes with NaN area assignments will have NaN values.
        """
        return self.node_model_df[self.area_column].to_dict()

    def get_border_lines_in_topological_up_and_down_direction(self, border_id: str) -> tuple[list[Hashable], list[Hashable]]:
        """Get transmission lines for a border classified by topological direction.

        This method identifies which transmission lines connect the two areas of a border
        and classifies them based on their topological direction relative to the border.
        
        Border directionality logic:
        - "Up" direction: Lines where node_from is in area_from and node_to is in area_to
        - "Down" direction: Lines where node_from is in area_to and node_to is in area_from
        
        This classification is essential for correctly aggregating directional quantities
        like power flows, where the sign and direction matter for market analysis.
        
        Args:
            border_id: Border identifier (e.g., 'DE-FR') that will be decomposed
                into area_from and area_to using the naming convention.
                
        Returns:
            Tuple containing two lists:
            - lines_up: Line IDs for lines in the "up" direction
            - lines_down: Line IDs for lines in the "down" direction
            
        Example:
            >>> # For border 'DE-FR'
            >>> lines_up, lines_down = calculator.get_border_lines_in_topological_up_and_down_direction('DE-FR')
            >>> # lines_up: Lines from German nodes to French nodes  
            >>> # lines_down: Lines from French nodes to German nodes
        """
        area_from, area_to = self.decompose_area_border_name_to_areas(border_id)
        nodes_in_area_from = self.node_model_df.loc[self.node_model_df[self.area_column] == area_from].index.to_list()
        nodes_in_area_to = self.node_model_df.loc[self.node_model_df[self.area_column] == area_to].index.to_list()
        lines_up = self.line_model_df.loc[
                self.line_model_df[self.node_from_col].isin(nodes_in_area_from)
                & self.line_model_df[self.node_to_col].isin(nodes_in_area_to)
            ].index.to_list()
        lines_down = self.line_model_df.loc[
                self.line_model_df[self.node_from_col].isin(nodes_in_area_to)
                & self.line_model_df[self.node_to_col].isin(nodes_in_area_from)
            ].index.to_list()
        return lines_up, lines_down

    @abstractmethod
    def calculate(self, **kwargs) -> pd.DataFrame:
        """Calculate the border variable. Must be implemented by subclasses.
        
        This method should contain the specific logic for aggregating line-level
        data to border level for the particular variable type. The implementation
        will vary based on the variable (flows, capacities, prices, etc.) and
        should handle directional aggregation appropriately.
        
        Args:
            **kwargs: Variable-specific parameters for the calculation
            
        Returns:
            DataFrame with border-level aggregated data. Index should be datetime
            for time series data, columns should be border identifiers.
            
        Raises:
            NotImplementedError: This is an abstract method
        """
        pass

    @property
    @abstractmethod
    def variable_name(self) -> str:
        """Name of the variable being calculated.
        
        This property should return a descriptive name for the variable being
        calculated by this calculator. Used for naming output columns and logging.
        
        Returns:
            String name of the variable (e.g., 'border_flow', 'border_capacity')
        """
        pass

    def _validate_time_series_data(self, df: pd.DataFrame, data_name: str):
        """Validate that time series data has appropriate datetime index.
        
        Logs warnings if the data doesn't have a DatetimeIndex, which may indicate
        data formatting issues or non-time-series data being used inappropriately.
        
        Args:
            df: DataFrame to validate
            data_name: Descriptive name of the data for logging purposes
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.warning(f"{data_name} does not have DatetimeIndex")
