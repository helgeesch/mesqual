from abc import ABC, abstractmethod
import pandas as pd

from mescal.utils.logging import get_logger

logger = get_logger(__name__)


class AreaVariableCalculatorBase(ABC):
    """Abstract base class for area-level variable calculators."""
    
    def __init__(self, node_model_df: pd.DataFrame, area_column: str):
        self.node_model_df = node_model_df
        self.area_column = area_column
        self.node_to_area_map = self._create_node_to_area_map()
        self.areas = sorted(self.node_model_df[area_column].dropna().unique())
        self._validate_inputs()
    
    def _validate_inputs(self):
        if self.area_column not in self.node_model_df.columns:
            raise ValueError(f"Column '{self.area_column}' not found in node_model_df")
    
    def _create_node_to_area_map(self) -> dict[str, str]:
        return self.node_model_df[self.area_column].to_dict()
    
    def get_area_nodes(self, area: str) -> list[str]:
        """Get all nodes belonging to a specific area."""
        return self.node_model_df[self.node_model_df[self.area_column] == area].index.tolist()
    
    @abstractmethod
    def calculate(self, **kwargs) -> pd.DataFrame:
        """Calculate the area variable. Must be implemented by subclasses."""
        pass
    
    def _validate_node_data(self, node_df: pd.DataFrame, data_name: str):
        """Validate that nodes are present in node_model_df."""
        missing_nodes = set(node_df.columns) - set(self.node_model_df.index)
        if missing_nodes:
            logger.warning(f"{len(missing_nodes)} nodes missing in node_model_df from {data_name}")
