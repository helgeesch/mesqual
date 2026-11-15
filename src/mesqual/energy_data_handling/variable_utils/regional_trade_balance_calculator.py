from typing import Literal, get_args

import pandas as pd
import networkx as nx

from mesqual.energy_data_handling.network_lines_data import NetworkLineFlowsData
from mesqual.energy_data_handling.variable_utils.trade_balance_frame import (
    EXP_VAR,
    IMP_VAR,
    NET_EXP_VAR,
    TOTAL_VAR,
    trade_balance_frame,
)

LossMode = Literal['at_own_end', 'excluded']


class RegionalTradeBalanceCalculator:
    """Aggregates bidirectional power flows between regions based on node-level flow data.

    Takes line-level flow data with bidirectional flows (up/down) and losses, and aggregates
    them to a higher regional level (e.g., countries, market areas) or keeps them at node level.
    Uses networkx for identifying region/node connections and handles multiple lines between
    the same region/node pairs.

    Loss modes (argument `losses` of `get_trade_balance`):

    - `'at_own_end'` (default): each region counts the flow at its own end of the lines. It
      exports the sent flow and imports the received flow, so the losses are inside the trade
      balance. `net_exp(A, B) + net_exp(B, A)` equals the losses on the border.
    - `'excluded'`: both regions use the sent flows, so the losses are not in the trade balance.
      `net_exp(A, B) = -net_exp(B, A)`.

    When agg_region_column is None, each node is treated as its own region, allowing for
    node-to-node trade balance analysis without aggregation.

    Raises:
        KeyError: A line end node (`node_from_col` or `node_to_col` of the line model) is
            empty in the line model, has no row in the node model, or has no value in
            `agg_region_column`. The message gives each reason with its nodes and the lines
            that reference them.

    Examples:
        >>> from mesqual.energy_data_handling import NetworkLineFlowsData
        >>> flows = NetworkLineFlowsData.from_net_flow_without_losses(dataset.fetch('Line.Results.flow_sent_net'))
        >>> calculator = RegionalTradeBalanceCalculator(dataset.fetch('Line.Model'), dataset.fetch('Node.Model'), 'country')
        >>> calculator.get_trade_balance(flows).xs('net_exp', axis=1, level=-1).sort_index(axis=1).iloc[:2].round(1)
        primary_country               C1     C2
        partner_country               C2     C1
        time
        2024-03-28 00:00:00+01:00 -345.7  345.7
        2024-03-28 01:00:00+01:00 -547.8  547.8
    """

    EXP_VAR = EXP_VAR
    IMP_VAR = IMP_VAR
    NET_EXP_VAR = NET_EXP_VAR
    TOTAL_VAR = TOTAL_VAR
    ALL_VARS = [EXP_VAR, IMP_VAR, NET_EXP_VAR, TOTAL_VAR]
    LOSS_MODES = get_args(LossMode)

    def __init__(
        self,
        line_model_df: pd.DataFrame,
        node_model_df: pd.DataFrame,
        agg_region_column: str | None = 'country',
        node_from_col: str = 'node_from',
        node_to_col: str = 'node_to',
    ):
        self.line_model_df = line_model_df
        self.node_model_df = node_model_df
        self.agg_region_column = agg_region_column
        self.node_from_col = node_from_col
        self.node_to_col = node_to_col
        self.node_to_agg_region_map = self._create_node_to_region_map()
        line_ends = self.line_model_df[[self.node_from_col, self.node_to_col]]
        self._line_regions = self._map_line_ends_to_regions(line_ends)
        self._check_line_end_nodes(line_ends)
        self.agg_region_graph = self._create_region_graph()

    def _create_node_to_region_map(self) -> dict:
        if self.agg_region_column is None:
            return {node: node for node in self.node_model_df.index}
        return self.node_model_df[self.agg_region_column].to_dict()

    def _map_line_ends_to_regions(self, line_ends: pd.DataFrame) -> pd.DataFrame:
        return line_ends.apply(lambda col: col.map(self.node_to_agg_region_map))

    def _check_line_end_nodes(self, line_ends: pd.DataFrame) -> None:
        empty = line_ends.isna()
        nodes_without_region = pd.concat(
            [line_ends[col][self._line_regions[col].isna() & ~empty[col]] for col in line_ends]
        )
        in_node_model = nodes_without_region.isin(self.node_model_df.index)
        reasons = []
        empty_lines = empty.any(axis=1)
        if empty_lines.any():
            lines = line_ends.index[empty_lines].tolist()
            reasons.append(
                f'the line model has an empty end node in {self.node_from_col!r} or {self.node_to_col!r} '
                f'(lines {lines})'
            )
        reasons += self._node_reasons(nodes_without_region[~in_node_model], 'have no row in the node model')
        reasons += self._node_reasons(
            nodes_without_region[in_node_model], f'have no value in column {self.agg_region_column!r} of the node model'
        )
        if reasons:
            raise KeyError(f'Invalid line end nodes: {"; ".join(reasons)}.')

    @staticmethod
    def _node_reasons(nodes_per_line: pd.Series, reason: str) -> list[str]:
        if nodes_per_line.empty:
            return []
        nodes = list(dict.fromkeys(nodes_per_line.tolist()))
        lines = list(dict.fromkeys(nodes_per_line.index.tolist()))
        return [f'nodes {nodes} {reason} (lines {lines})']

    def _create_region_graph(self) -> nx.Graph:
        graph = nx.Graph()
        for region_from, region_to in self._line_regions.itertuples(index=False):
            if region_from != region_to and not graph.has_edge(region_from, region_to):
                graph.add_edge(region_from, region_to)
        return graph

    def _get_net_exp_for_couple(
        self, primary, secondary, flow_data: NetworkLineFlowsData, losses: LossMode
    ) -> pd.Series:
        region_from = self._line_regions[self.node_from_col]
        region_to = self._line_regions[self.node_to_col]
        mask_forward = (region_from == primary) & (region_to == secondary)
        mask_backward = (region_from == secondary) & (region_to == primary)

        lines_forward = self.line_model_df[mask_forward].index
        lines_backward = self.line_model_df[mask_backward].index

        if losses == 'at_own_end':
            fwd_in, fwd_out = flow_data.sent_up, flow_data.received_down
            bwd_in, bwd_out = flow_data.sent_down, flow_data.received_up
        else:
            fwd_in, fwd_out = flow_data.sent_up, flow_data.sent_down
            bwd_in, bwd_out = flow_data.sent_down, flow_data.sent_up

        net = (
            fwd_in[lines_forward].sum(axis=1)
            - fwd_out[lines_forward].sum(axis=1)
            + bwd_in[lines_backward].sum(axis=1)
            - bwd_out[lines_backward].sum(axis=1)
        )

        any_value = (
            fwd_in[lines_forward].notna().any(axis=1)
            | fwd_out[lines_forward].notna().any(axis=1)
            | bwd_in[lines_backward].notna().any(axis=1)
            | bwd_out[lines_backward].notna().any(axis=1)
        )
        return net.where(any_value)

    def get_trade_balance(self, flow_data: NetworkLineFlowsData, losses: LossMode = 'at_own_end') -> pd.DataFrame:
        """Computes the trade balance of each region with each neighbouring region.

        Args:
            flow_data: Line flows with sent and received values per direction.
            losses: How line losses enter the trade balance: `'at_own_end'` or `'excluded'`.
                See the class docstring.

        Returns:
            DataFrame with three column levels: primary region, partner region and variable
                (`exp`, `imp`, `net_exp`, `total`). `net_exp` is positive for flows from the
                primary region to the partner.

        Raises:
            ValueError: If `losses` is not one of `LOSS_MODES`.
        """
        if losses not in self.LOSS_MODES:
            raise ValueError(f'losses must be one of {self.LOSS_MODES}, got {losses!r}.')
        net_exports = {
            (primary, secondary): self._get_net_exp_for_couple(primary, secondary, flow_data, losses)
            for primary in self.get_all_regions()
            for secondary in self.get_region_neighbors(primary)
        }
        return trade_balance_frame(net_exports, self.primary_name, self.partner_name, flow_data.sent_up.index)

    def get_region_neighbors(self, region: str) -> set:
        return set(self.agg_region_graph.neighbors(region))

    def get_all_regions(self) -> set:
        return set(self.agg_region_graph.nodes())

    @property
    def primary_name(self) -> str:
        return f'primary_{self.agg_region_column}'

    @property
    def partner_name(self) -> str:
        return f'partner_{self.agg_region_column}'

    def aggregate_trade_balance_to_primary_level(self, trade_balance_df: pd.DataFrame) -> pd.DataFrame:
        """Reduces three-level trade balance DataFrame to primary region and variable only."""
        if trade_balance_df.columns.nlevels != 3:
            raise ValueError('Input DataFrame must have three column levels')
        if trade_balance_df.columns.names != [self.primary_name, self.partner_name, 'variable']:
            raise ValueError('Input DataFrame must be in format from aggregate_flows')

        return trade_balance_df.T.groupby(level=[self.primary_name, 'variable']).sum(min_count=1).T

    def get_net_position_per_primary_level(self, trade_balance_df: pd.DataFrame) -> pd.DataFrame:
        return self.aggregate_trade_balance_to_primary_level(trade_balance_df).xs(self.NET_EXP_VAR, level=-1, axis=1)
