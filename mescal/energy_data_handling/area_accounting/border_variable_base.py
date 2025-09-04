from abc import ABC, abstractmethod
from typing import Hashable

import pandas as pd

from mescal.energy_data_handling.area_accounting.border_model_generator import AreaBorderNamingConventions
from mescal.utils.logging import get_logger

logger = get_logger(__name__)


class AreaBorderVariableCalculatorBase(ABC, AreaBorderNamingConventions):
    """Abstract base class for calculating energy variables at area border level.
    
    This base class provides functionality for aggregating line-level energy data 
    (flows, capacities, prices) to area border level. An area border represents 
    the transmission interface between two areas (countries, bidding zones, etc.).
    
    The class handles the complex mapping from transmission lines to area borders,
    including proper handling of line directionality. Lines are classified as either
    "up" or "down" relative to the border direction based on their node endpoints.
    
    Energy market context:
    In electricity markets, transmission lines connect nodes within and between 
    different market areas. Border variables represent aggregated quantities at 
    the interface between areas, which are crucial for:
    - Cross-border flow analysis
    - Market coupling calculations  
    - Congestion rent allocation
    - Transmission capacity planning
    - Price spread analysis between areas
    
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
            
        Raises:
            NotImplementedError: This is an abstract property
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


if __name__ == "__main__":
    # Example implementation of the abstract base class
    class ExampleBorderCapacityCalculator(AreaBorderVariableCalculatorBase):
        """Example implementation that calculates border capacities."""
        
        @property 
        def variable_name(self) -> str:
            return "example_capacity"
            
        def calculate(self, line_capacities: pd.DataFrame) -> pd.DataFrame:
            """Sum line capacities for each border."""
            self._validate_time_series_data(line_capacities, "line_capacities")
            
            border_results = {}
            
            for border_id in self.area_border_model_df.index:
                lines_up, lines_down = self.get_border_lines_in_topological_up_and_down_direction(border_id)
                all_border_lines = lines_up + lines_down
                
                if all_border_lines:
                    # Filter to lines that exist in capacity data
                    available_lines = [line for line in all_border_lines if line in line_capacities.columns]
                    if available_lines:
                        border_results[border_id] = line_capacities[available_lines].sum(axis=1)
                    else:
                        # Create empty series if no capacity data available
                        border_results[border_id] = pd.Series(index=line_capacities.index, dtype=float)
                else:
                    # No lines found for this border
                    border_results[border_id] = pd.Series(index=line_capacities.index, dtype=float)
            
            result_df = pd.DataFrame(border_results)
            result_df.columns.name = self.border_identifier
            return result_df
    
    # Create example data structures
    # Node model with area assignments
    node_model_df = pd.DataFrame({
        'country': ['DE', 'DE', 'FR', 'FR', 'BE', 'BE'],
        'voltage': [380, 220, 380, 220, 380, 220]
    }, index=['DE1', 'DE2', 'FR1', 'FR2', 'BE1', 'BE2'])
    
    # Line model with interconnections
    line_model_df = pd.DataFrame({
        'node_from': ['DE1', 'DE2', 'FR1', 'FR2', 'BE1'],
        'node_to': ['FR1', 'FR2', 'BE1', 'BE2', 'DE1'],  # Lines connecting different countries
        'voltage': [380, 220, 380, 220, 380],
        'length_km': [500, 400, 300, 250, 600]
    }, index=['DE-FR1', 'DE-FR2', 'FR-BE1', 'FR-BE2', 'BE-DE1'])
    
    # Area border model (borders between countries)
    area_border_model_df = pd.DataFrame({
        'capacity_limit': [2000, 1500, 1200]  # MW
    }, index=['DE-FR', 'FR-BE', 'BE-DE'])
    
    print("Node model (countries):")
    print(node_model_df)
    print("\nLine model (interconnections):")
    print(line_model_df)
    print("\nBorder model:")
    print(area_border_model_df)
    
    # Initialize the calculator
    calculator = ExampleBorderCapacityCalculator(
        area_border_model_df=area_border_model_df,
        line_model_df=line_model_df,
        node_model_df=node_model_df,
        area_column='country'
    )
    
    # Test border line identification
    print(f"\nBorder line direction analysis:")
    for border_id in area_border_model_df.index:
        lines_up, lines_down = calculator.get_border_lines_in_topological_up_and_down_direction(border_id)
        print(f"Border {border_id}:")
        print(f"  - Lines up: {lines_up}")
        print(f"  - Lines down: {lines_down}")
        
        # Show which areas these correspond to
        area_from, area_to = calculator.decompose_area_border_name_to_areas(border_id)
        print(f"  - Direction: {area_from} → {area_to} (up), {area_to} → {area_from} (down)")
    
    # Create example capacity time series
    import numpy as np
    time_index = pd.date_range('2024-01-01', periods=24, freq='h')
    line_capacities = pd.DataFrame(
        np.random.uniform(800, 1200, size=(24, len(line_model_df))),
        index=time_index,
        columns=line_model_df.index
    )
    
    print(f"\nLine capacities (first 5 hours):")
    print(line_capacities.head())
    
    # Calculate border capacities
    border_capacities = calculator.calculate(line_capacities)
    print(f"\nBorder capacities (first 5 hours):")
    print(border_capacities.head())
    
    # Demonstrate validation
    print(f"\nValidator test - Non-datetime index:")
    non_datetime_data = pd.DataFrame(
        np.random.uniform(800, 1200, size=(3, len(line_model_df))),
        index=[1, 2, 3],  # Non-datetime index
        columns=line_model_df.index
    )
    result_with_warning = calculator.calculate(non_datetime_data)
