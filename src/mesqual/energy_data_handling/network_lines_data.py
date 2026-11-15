from dataclasses import dataclass

import pandas as pd


@dataclass(eq=False)
class NetworkLineFlowsData:
    """Wrapper for bidirectional flow data of network transmission lines.

    The four frames hold the flows of the lines in both directions, before and
    after losses. They have one column per line and the same index.

    Flow direction conventions:

    - sent_up: Flow that enters the line at node_from (towards node_to)
    - received_up: Flow that leaves the line at node_to after losses (coming from node_from)
    - sent_down: Flow that enters the line at node_to (towards node_from)
    - received_down: Flow that leaves the line at node_from after losses (coming from node_to)

    The distinction between 'sent' and 'received' allows for modeling transmission
    losses, where received_flow = sent_flow * (1 - loss_rate).

    Args:
        sent_up: DataFrame with the flow that enters at node_from
        received_up: DataFrame with the flow received at node_to after losses
        sent_down: DataFrame with the flow that enters at node_to
        received_down: DataFrame with the flow received at node_from after losses

    Raises:
        ValueError: If indices or columns of the four DataFrames don't match
    """

    sent_up: pd.DataFrame
    received_up: pd.DataFrame
    sent_down: pd.DataFrame
    received_down: pd.DataFrame

    def __post_init__(self):
        """Validate that all DataFrames have matching indices and columns.

        Raises:
            ValueError: If any DataFrame has mismatched indices or columns.
        """
        others = {
            'received_up': self.received_up,
            'sent_down': self.sent_down,
            'received_down': self.received_down,
        }
        for name, df in others.items():
            if not self.sent_up.index.equals(df.index):
                raise ValueError(f'Index mismatch: sent_up vs {name}')
            if not self.sent_up.columns.equals(df.columns):
                raise ValueError(f'Columns mismatch: sent_up vs {name}')

    @classmethod
    def from_net_flow_without_losses(cls, net_flow: pd.DataFrame) -> 'NetworkLineFlowsData':
        """Create NetworkLineFlowsData from net line flow data assuming no transmission losses.

        Converts net flow data (where positive values indicate flow in up direction
        and negative values indicate flow in down direction) into the bidirectional
        flow representation used by this class.

        Args:
            net_flow: DataFrame with net flow values. Positive = up direction,
                negative = down direction

        Returns:
            NetworkLineFlowsData instance with flows split into up/down directions

        Examples:
            >>> flows = NetworkLineFlowsData.from_net_flow_without_losses(pd.DataFrame({'L1': [100.0, -50.0]}))
            >>> flows.sent_up['L1'].tolist(), flows.sent_down['L1'].tolist()
            ([100.0, 0.0], [-0.0, 50.0])
        """
        positive_flow = net_flow.clip(lower=0)
        negative_flow = -net_flow.clip(upper=0)

        return cls(
            sent_up=positive_flow, received_up=positive_flow, sent_down=negative_flow, received_down=negative_flow
        )

    @classmethod
    def from_nodal_net_injection(
        cls, node_a_net_injection: pd.DataFrame, node_b_net_injection: pd.DataFrame
    ) -> 'NetworkLineFlowsData':
        """Create NetworkLineFlowsData from nodal net-injection data.

        Converts nodal net-injection data (where positive values indicate flow from node towards line (injection)
        and negative values indicate flow from line towards node (ejection)) into the bidirectional
        flow representation used by this class. Automatically computes losses.
        Node_A refers to the topological "node_from" and
        Node_B to the topological "node_to" in the definition of the lines.

        Args:
            node_a_net_injection: DataFrame with net injection at Node_A.
                positive = injection, negative = ejection
            node_b_net_injection: DataFrame with net injection at Node_B.
                positive = injection, negative = ejection

        Returns:
            NetworkLineFlowsData instance with flows split into up/down directions

        Examples:
            >>> flows = NetworkLineFlowsData.from_nodal_net_injection(
            ...     pd.DataFrame({'L1': [100.0, -49.0]}), pd.DataFrame({'L1': [-98.0, 50.0]})
            ... )
            >>> flows.sent_up['L1'].tolist(), flows.received_up['L1'].tolist()  # 2 MW losses in the first step
            ([100.0, 0.0], [98.0, -0.0])
        """

        return cls(
            sent_up=node_a_net_injection.clip(0),
            received_up=-1 * node_b_net_injection.clip(None, 0),
            sent_down=node_b_net_injection.clip(0),
            received_down=-1 * node_a_net_injection.clip(None, 0),
        )

    @classmethod
    def from_up_and_down_flow_without_losses(
        cls, flow_up: pd.DataFrame, flow_down: pd.DataFrame
    ) -> 'NetworkLineFlowsData':
        """Create NetworkLineFlowsData from separate up and down flow data without losses.

        This constructor assumes that there are no transmission losses, so sent and
        received flows are identical in each direction.

        Args:
            flow_up: DataFrame with flow data in up direction (node_from -> node_to)
            flow_down: DataFrame with flow data in down direction (node_to -> node_from)

        Returns:
            NetworkLineFlowsData instance where sent and received flows are equal

        Examples:
            >>> flows = NetworkLineFlowsData.from_up_and_down_flow_without_losses(
            ...     pd.DataFrame({'L1': [100.0, 80.0]}), pd.DataFrame({'L1': [50.0, 60.0]})
            ... )
            >>> flows.received_down['L1'].tolist()
            [50.0, 60.0]
        """
        return cls(sent_up=flow_up, received_up=flow_up, sent_down=flow_down, received_down=flow_down)
