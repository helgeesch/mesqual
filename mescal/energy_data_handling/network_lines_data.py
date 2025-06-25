from dataclasses import dataclass

import pandas as pd


class NetworkLineFlowsData:
    """
    Wrapper for flow data of lines.

    Conventions:
        sent_up: Flow entering line at node_from (towards node_to)
        received_up: Flow leaving line at node_from after losses (coming from node_to)
        sent_down: Flow entering line at node_to (towards node_from)
        received_down: Flow leaving line at node_to after losses (coming from node_from)
    """
    def __init__(
            self,
            sent_up: pd.DataFrame,
            received_up: pd.DataFrame,
            sent_down: pd.DataFrame,
            received_down: pd.DataFrame,
            granularity: None | float | pd.Series = None
    ):
        self.sent_up = sent_up
        self.received_up = received_up
        self.sent_down = sent_down
        self.received_down = received_down
        self.granularity = granularity
        self.__post_init__()

    def __post_init__(self):
        for s in [self.received_up, self.sent_down, self.received_down]:
            if not self.sent_up.index.equals(s.index):
                raise ValueError(f'All indices must be equal!')
            if not self.sent_up.columns.equals(s.columns):
                raise ValueError(f'All columns must be equal!')

    def from_mw_to_mwh(self) -> 'NetworkLineFlowsData':
        # TODO
        raise NotImplementedError

    def from_mwh_to_mw(self) -> 'NetworkLineFlowsData':
        # TODO
        raise NotImplementedError

    @classmethod
    def from_net_flow_without_losses(cls, net_flow: pd.DataFrame) -> "NetworkLineFlowsData":
        positive_flow = net_flow.clip(lower=0)
        negative_flow = -net_flow.clip(upper=0)

        return cls(
            sent_up=positive_flow,
            received_up=positive_flow,
            sent_down=negative_flow,
            received_down=negative_flow
        )

    @classmethod
    def from_up_and_down_flow_without_losses(
            cls,
            flow_up: pd.DataFrame,
            flow_down: pd.DataFrame
    ) -> "NetworkLineFlowsData":
        return cls(
            sent_up=flow_up,
            received_up=flow_up,
            sent_down=flow_down,
            received_down=flow_down
        )


@dataclass
class NetworkLineCapacitiesData:
    """
    Wrapper for capacity data of lines.

    Conventions:
        capacities_up: Capacities of line in up direction (node_from -> node_to)
        capacities_down: Capacities of line in down direction (node_to -> node_from)
    """
    capacities_up: pd.DataFrame
    capacities_down: pd.DataFrame
    granularity: None | float | pd.Series = None

    def __post_init__(self):
        for s in [self.capacities_down]:
            if not self.capacities_up.index.equals(s.index):
                raise ValueError(f'All indices must be equal!')
            if not self.capacities_up.columns.equals(s.columns):
                raise ValueError(f'All columns must be equal!')

    def from_mw_to_mwh(self) -> 'NetworkLineCapacitiesData':
        # TODO
        raise NotImplementedError

    def from_mwh_to_mw(self) -> 'NetworkLineCapacitiesData':
        # TODO
        raise NotImplementedError

    @classmethod
    def from_symmetric_capacities(cls, capacities: pd.DataFrame) -> "NetworkLineCapacitiesData":
        return cls(capacities, capacities)
