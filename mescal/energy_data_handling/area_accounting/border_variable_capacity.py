from typing import Literal
from dataclasses import dataclass

import numpy as np
import pandas as pd

from mescal.energy_data_handling.network_lines_data import NetworkLineCapacitiesData
from mescal.energy_data_handling.area_border.border_variable_base import AreaBorderVariableCalculatorBase


class BorderCapacityCalculator(AreaBorderVariableCalculatorBase):
    """Calculates aggregated capacities for area borders."""

    @property
    def variable_name(self) -> str:
        return "border_capacity"

    def calculate(
            self,
            line_capacity_data: NetworkLineCapacitiesData,
            direction: Literal['up', 'down'] = 'up'
    ) -> pd.DataFrame:
        """Aggregate line capacities to area border level."""
        border_capacities = {}

        for border_id, border in self.area_border_model_df.iterrows():
            lines_up, lines_down = self.get_border_lines_in_topological_up_and_down_direction(border_id)

            if not lines_up and not lines_down:
                continue

            if direction == 'up':
                capacities_up_lines = line_capacity_data.capacities_up[lines_up]
                capacities_down_lines = line_capacity_data.capacities_down[lines_down]
            elif direction == 'down':
                capacities_up_lines = line_capacity_data.capacities_down[lines_up]
                capacities_down_lines = line_capacity_data.capacities_up[lines_down]
            else:
                raise ValueError(f"Unknown capacity direction: {direction}")

            capacity = capacities_up_lines.sum(axis=1) + capacities_down_lines.sum(axis=1)
            if capacities_up_lines.empty and capacities_down_lines.empty:
                capacity.values[:] = np.nan
            else:
                capacity[capacities_up_lines.isna().all(axis=1) & capacities_down_lines.isna().all(axis=1)] = np.nan

            border_capacities[border_id] = capacity

        result = pd.DataFrame(border_capacities)
        result.columns.name = self.border_identifier
        return result
