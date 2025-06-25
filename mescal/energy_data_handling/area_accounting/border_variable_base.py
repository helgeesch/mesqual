from abc import ABC, abstractmethod
from typing import Hashable

import pandas as pd

from mescal.energy_data_handling.area_border.border_model_generator import AreaBorderNamingConventions
from mescal.utils.logging import get_logger

logger = get_logger(__name__)


class AreaBorderVariableCalculatorBase(ABC, AreaBorderNamingConventions):
    """Abstract base class for area border variable calculators."""

    def __init__(
        self,
        area_border_model_df: pd.DataFrame,
        line_model_df: pd.DataFrame,
        node_model_df: pd.DataFrame,
        area_column: str,
        node_from_col: str = 'node_from',
        node_to_col: str = 'node_to'
    ):
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
        if self.area_column not in self.node_model_df.columns:
            raise ValueError(f"Column '{self.area_column}' not found in node_model_df")
        if self.node_from_col not in self.line_model_df.columns:
            raise ValueError(f"Column '{self.node_from_col}' not found in line_model_df")
        if self.node_to_col not in self.line_model_df.columns:
            raise ValueError(f"Column '{self.node_to_col}' not found in line_model_df")

    def _create_node_to_area_map(self) -> dict[Hashable, str]:
        return self.node_model_df[self.area_column].to_dict()

    def get_border_lines_in_topological_up_and_down_direction(self, border_id: str) -> tuple[list[Hashable], list[Hashable]]:
        """Get all lines belonging to a specific border including the topological direction.

        Direction "up" means that node_from is in area_from and node_to is in area_to;
        Direction "down" means that node_to is in area_from and node_from is in area_to.
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
        """Calculate the border variable. Must be implemented by subclasses."""
        pass

    @property
    @abstractmethod
    def variable_name(self) -> str:
        """Name of the variable being calculated."""
        pass

    def _validate_time_series_data(self, df: pd.DataFrame, data_name: str):
        """Validate time series data has datetime index."""
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.warning(f"{data_name} does not have DatetimeIndex")
