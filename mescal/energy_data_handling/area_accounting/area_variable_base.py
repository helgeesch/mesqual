from abc import ABC, abstractmethod
import pandas as pd

from mescal.utils.logging import get_logger

logger = get_logger(__name__)


class AreaVariableCalculatorBase(ABC):
    """Abstract base class for calculating energy variables aggregated at area level.
    
    This base class provides common functionality for aggregating node-level energy data
    (such as generation, demand, prices) to higher-level areas (countries, bidding zones,
    market areas). It handles the mapping between nodes and areas and provides validation
    and utility methods for area-based calculations.
    
    The class is designed to be subclassed for specific variable types, with each subclass
    implementing its own calculation logic while leveraging the common area mapping and
    validation functionality provided here.
    
    Energy market context:
    In electricity markets, many variables are naturally defined at the nodal level 
    (generators, loads, prices) but need to be aggregated to market or geographical
    areas for analysis, reporting, and trading. This aggregation must handle missing
    data, different node counts per area, and preserve energy-specific semantics.
    
    Args:
        node_model_df: DataFrame containing node information with area assignments.
            Index should be node identifiers, must contain the specified area_column.
        area_column: Name of the column in node_model_df that contains area assignments.
            Each node should be assigned to exactly one area (NaN values are allowed).
            
    Attributes:
        node_model_df: The input node model DataFrame
        area_column: Name of the area assignment column
        node_to_area_map: Dictionary mapping node IDs to area names
        areas: Sorted list of unique area names (excluding NaN)
        
    Raises:
        ValueError: If area_column is not found in node_model_df
        
    Example:

        >>> import pandas as pd
        >>> # Node model with area assignments
        >>> node_model = pd.DataFrame({
        ...     'country': ['DE', 'DE', 'FR', 'FR', 'BE'],
        ...     'voltage': [380, 220, 380, 220, 380]
        ... }, index=['DE1', 'DE2', 'FR1', 'FR2', 'BE1'])
        >>> 
        >>> # Subclass implementation
        >>> class MyAreaCalculator(AreaVariableCalculatorBase):
        ...     def calculate(self, **kwargs):
        ...         return pd.DataFrame()  # Implementation here
        >>> 
        >>> calculator = MyAreaCalculator(node_model, 'country')
        >>> print(calculator.areas)  # ['BE', 'DE', 'FR']
        >>> print(calculator.get_area_nodes('DE'))  # ['DE1', 'DE2']
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
        self.node_to_area_map = self._create_node_to_area_map()
        self.areas = sorted(self.node_model_df[area_column].dropna().unique())
        self._validate_inputs()
    
    def _validate_inputs(self):
        """Validate input parameters during initialization.
        
        Raises:
            ValueError: If area_column is not found in node_model_df
        """
        if self.area_column not in self.node_model_df.columns:
            raise ValueError(f"Column '{self.area_column}' not found in node_model_df")
    
    def _create_node_to_area_map(self) -> dict[str, str]:
        return self.node_model_df[self.area_column].to_dict()
    
    def get_area_nodes(self, area: str) -> list[str]:
        """Get all nodes belonging to a specific area.
        
        Args:
            area: Area name to get nodes for
            
        Returns:
            List of node IDs that belong to the specified area
            
        Example:

            >>> calculator = MyAreaCalculator(node_model, 'country')
            >>> german_nodes = calculator.get_area_nodes('DE')
            >>> print(german_nodes)  # ['DE1', 'DE2']
        """
        return self.node_model_df[self.node_model_df[self.area_column] == area].index.tolist()
    
    @abstractmethod
    def calculate(self, **kwargs) -> pd.DataFrame:
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
            
        Raises:
            NotImplementedError: This is an abstract method
        """
        pass
    
    def _validate_node_data(self, node_df: pd.DataFrame, data_name: str):
        """Validate that required nodes are present in node_model_df.
        
        Logs warnings for any nodes found in the data that are not in the node model.
        This is important for detecting data inconsistencies or model updates.
        
        Args:
            node_df: DataFrame containing node-level data to validate
            data_name: Descriptive name of the data being validated (for logging)
            
        Example:

            >>> # Log warning if generation_data has nodes not in node_model_df
            >>> calculator._validate_node_data(generation_data, "generation")
        """
        missing_nodes = set(node_df.columns) - set(self.node_model_df.index)
        if missing_nodes:
            logger.warning(f"{len(missing_nodes)} nodes missing in node_model_df from {data_name}")


if __name__ == "__main__":
    # Example implementation of the abstract base class
    class ExampleSumCalculator(AreaVariableCalculatorBase):
        """Example implementation that sums node-level data to area level."""
        
        def calculate(self, node_data: pd.DataFrame) -> pd.DataFrame:
            """Sum node data for each area (extensive variable aggregation)."""
            self._validate_node_data(node_data, "example_data")
            
            result_dict = {}
            for area in self.areas:
                area_nodes = self.get_area_nodes(area)
                # Filter to nodes that exist in both model and data
                available_nodes = [n for n in area_nodes if n in node_data.columns]
                if available_nodes:
                    result_dict[area] = node_data[available_nodes].sum(axis=1)
                else:
                    # Create empty series with same index if no data available
                    result_dict[area] = pd.Series(index=node_data.index, dtype=float)
            
            result_df = pd.DataFrame(result_dict)
            result_df.columns.name = self.area_column
            return result_df
    
    # Create example node model
    node_model_df = pd.DataFrame({
        'country': ['DE', 'DE', 'FR', 'FR', 'BE', None],  # Include NaN for testing
        'voltage': [380, 220, 380, 220, 380, 110],
        'market_area': ['DE_LU', 'DE_LU', 'FR', 'FR', 'BE', 'DE_LU']
    }, index=['DE1', 'DE2', 'FR1', 'FR2', 'BE1', 'DE3'])
    
    # Create example time series data (e.g., generation)
    import numpy as np
    time_index = pd.date_range('2024-01-01', periods=24, freq='h')
    node_data = pd.DataFrame(
        np.random.uniform(100, 1000, size=(24, len(node_model_df))),
        index=time_index,
        columns=node_model_df.index
    )
    
    print("Node model:")
    print(node_model_df)
    print("\nNode data (first 5 hours):")
    print(node_data.head())
    
    # Example 1: Country-level aggregation
    calculator_country = ExampleSumCalculator(node_model_df, 'country')
    print(f"\nIdentified areas (countries): {calculator_country.areas}")
    print(f"Nodes in Germany: {calculator_country.get_area_nodes('DE')}")
    
    country_aggregated = calculator_country.calculate(node_data)
    print("\nCountry-level aggregated data (first 5 hours):")
    print(country_aggregated.head())
    
    # Example 2: Market area aggregation
    calculator_market = ExampleSumCalculator(node_model_df, 'market_area')
    print(f"\nIdentified areas (market areas): {calculator_market.areas}")
    
    market_aggregated = calculator_market.calculate(node_data)
    print("\nMarket area aggregated data (first 5 hours):")
    print(market_aggregated.head())
