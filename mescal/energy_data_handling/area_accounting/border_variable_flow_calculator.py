from typing import Literal
from dataclasses import dataclass

import numpy as np
import pandas as pd

from mescal.energy_data_handling.network_lines_data import NetworkLineFlowsData
from mescal.energy_data_handling.area_border.border_variable_base import AreaBorderVariableCalculatorBase


class BorderFlowCalculator(AreaBorderVariableCalculatorBase):
    """Calculates aggregated flows for area borders."""
    
    @property
    def variable_name(self) -> str:
        return "border_flow"
    
    def calculate(
        self,
        line_flow_data: NetworkLineFlowsData,
        flow_type: Literal['sent', 'received'] = 'sent',
        direction: Literal['up', 'down', 'net'] = 'net'
    ) -> pd.DataFrame:
        """Aggregate line flows to area border level."""
        border_flows = {}
        
        for border_id, border in self.area_border_model_df.iterrows():
            lines_up, lines_down = self.get_border_lines_in_topological_up_and_down_direction(border_id)

            if not lines_up and not lines_down:
                continue

            if flow_type == 'sent':
                flows_up = line_flow_data.sent_up[lines_up]
                flows_down = line_flow_data.sent_down[lines_down]
            elif flow_type == 'received':
                flows_up = line_flow_data.received_up[lines_up]
                flows_down = line_flow_data.received_down[lines_down]
            else:
                raise ValueError(f"Unknown flow_type: {flow_type}")

            flow_up = flows_up.sum(axis=1)
            flow_up[flows_up.isna().all(axis=1)] = np.nan
            flow_down = flows_down.sum(axis=1)
            flow_down[flows_down.isna().all(axis=1)] = np.nan

            if direction == 'up':
                flow = flow_up
            elif direction == 'down':
                flow = flow_down
            elif direction == 'net':
                flow = flow_up - flow_down
                flow[flow_up.isna() & flow_down.isna()] = np.nan
            else:
                raise ValueError(f"Unknown flow direction: {direction}")

            border_flows[border_id] = flow
        
        result = pd.DataFrame(border_flows)
        result.columns.name = self.border_identifier
        return result
