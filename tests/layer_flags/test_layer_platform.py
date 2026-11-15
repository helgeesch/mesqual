"""Layer flags on the layer test platform: area flags, border flows, a trade balance on a border base, parameters and describe output."""

import pandas as pd
import pytest

from mesqual.datasets import INHERIT
from mesqual.energy_data_handling.area_accounting import AreaMembership
from mesqual.energy_data_handling.network_lines_data import NetworkLineFlowsData
from mesqual.energy_data_handling.variable_utils import RegionalTradeBalanceCalculator
from mesqual.enums import TopologyTypeEnum
from mesqual.layer_flags import edges_on_no_border
from mesqual.testing import DummyScenario
from tests.layer_flags.platform import LayerPlatform

COUNTRY_LINES_WITH = {'C1 - C2': ['L1', 'L4'], 'C2 - C1': ['L2']}
"""Lines from N1, N3 (C1) to N2, N4 (C2): L1 N1-N2, L4 N3-N4. L2 runs N2-N3. L3 and L5 are inside one country."""
MACROREGION_LINES_WITH = {'North - South': ['L1'], 'South - North': ['L2', 'L5']}
"""N4 lies in North: L4 is inside North, and L5 N2-N4 runs from South to North."""


@pytest.fixture(scope='module')
def ds():
    return LayerPlatform()


def lines_of(ds, area_column):
    """The lines per area pair from `AreaMembership.edge_areas`, as border names."""
    areas = AreaMembership.edge_areas(
        ds.fetch('Line.Model'), ds.fetch('Node.Model')[area_column], 'node_from', 'node_to'
    )
    pairs = areas[AreaMembership.crosses_border(areas)].groupby(['area_from', 'area_to']).groups
    return {f'{a} - {b}': sorted(lines) for (a, b), lines in pairs.items()}


def edge_counts(ds, border_layer):
    model = ds.fetch(f'{border_layer}.Model')
    return {b: (row.n_edges_with_border, row.n_edges_against_border) for b, row in model.iterrows()}


class TestAreaFlags:
    def test_area_model_counts_nodes(self, ds):
        assert ds.fetch('Country.Model')['node_count'].to_dict() == {'C1': 2, 'C2': 2}
        assert ds.fetch('Macroregion.Model')['node_count'].to_dict() == {'North': 3, 'South': 1}

    def test_area_price_is_demand_weighted_mean(self, ds):
        prices, demand = ds.fetch('Node.Results.price'), ds.fetch('Node.Results.demand_total')
        expected = (prices[['N1', 'N3']] * demand[['N1', 'N3']]).sum(axis=1) / demand[['N1', 'N3']].sum(axis=1)
        result = ds.fetch('Country.Results.price')
        pd.testing.assert_series_equal(result['C1'], expected, check_names=False)
        assert result.columns.name == 'country'

    def test_two_layers_on_the_same_nodes_give_different_areas(self, ds):
        country, macroregion = ds.fetch('Country.Results.price'), ds.fetch('Macroregion.Results.price')
        assert not country['C1'].equals(macroregion['North'])
        pd.testing.assert_series_equal(macroregion['South'], ds.fetch('Node.Results.price')['N2'], check_names=False)

    def test_price_spread_is_end_minus_start(self, ds):
        prices = ds.fetch('Country.Results.price')
        spread = ds.fetch('CountryBorder.Results.price_spread')
        pd.testing.assert_series_equal(spread['C1 - C2'], prices['C2'] - prices['C1'], check_names=False)
        pd.testing.assert_series_equal(spread['C2 - C1'], prices['C1'] - prices['C2'], check_names=False)

    def test_topology_and_unit_come_from_class_and_base(self, ds):
        index = ds.flag_index
        assert index.get_topology_type('Country.Results.price') is TopologyTypeEnum.Area
        assert index.get_topology_type('CountryBorder.Results.price_spread') is TopologyTypeEnum.Border
        assert index.get_spec('Country.Results.price').unit == index.get_spec('Node.Results.price').unit
        assert index.get_linked_model_flag('Country.Results.price') == 'Country.Model'


class TestBorderFlows:
    def test_edge_areas_sort_lines_by_border_direction(self, ds):
        assert lines_of(ds, 'country') == COUNTRY_LINES_WITH
        assert lines_of(ds, 'macroregion') == MACROREGION_LINES_WITH

    def test_border_model_counts_lines_per_border_direction(self, ds):
        assert edge_counts(ds, 'CountryBorder') == {'C1 - C2': (2, 1), 'C2 - C1': (1, 2)}
        assert edge_counts(ds, 'MacroregionBorder') == {'North - South': (1, 2), 'South - North': (2, 1)}

    @pytest.mark.parametrize('side', ['sent', 'received'])
    def test_border_flow_up_adds_lines_against_border_with_down(self, ds, side):
        up, down = ds.fetch(f'Line.Results.flow_{side}_up'), ds.fetch(f'Line.Results.flow_{side}_down')
        result = ds.fetch(f'CountryBorder.Results.flow_{side}_up')
        expected = up[['L1', 'L4']].sum(axis=1) + down[['L2']].sum(axis=1)
        pd.testing.assert_series_equal(result['C1 - C2'], expected, check_names=False)
        assert result.columns.name == 'country_border'

    def test_border_flow_net_negates_lines_against_border(self, ds):
        net = ds.fetch('Line.Results.flow_sent_net')
        result = ds.fetch('MacroregionBorder.Results.flow_sent_net')
        pd.testing.assert_series_equal(
            result['South - North'], net[['L2', 'L5']].sum(axis=1) - net['L1'], check_names=False
        )

    def test_border_flow_up_of_one_direction_is_down_of_the_other(self, ds):
        up = ds.fetch('CountryBorder.Results.flow_sent_up')
        down = ds.fetch('CountryBorder.Results.flow_sent_down')
        pd.testing.assert_series_equal(up['C1 - C2'], down['C2 - C1'], check_names=False)

    def test_border_capacity_swaps_lines_against_border(self, ds):
        up, down = ds.fetch('Line.Constraints.capacity_up'), ds.fetch('Line.Constraints.capacity_down')
        result = ds.fetch('CountryBorder.Constraints.capacity_down')
        expected = down[['L1', 'L4']].sum(axis=1) + up['L2']
        pd.testing.assert_series_equal(result['C1 - C2'], expected, check_names=False)

    def test_base_params_reach_the_line_flows(self, ds):
        flag = 'CountryBorder.Results.scaled_flow_up'
        pd.testing.assert_frame_equal(ds.fetch(flag, scale=2), 2 * ds.fetch(flag))
        record = ds.get_fetch_record(flag, scale=2)
        line_flows = [d for d in record.iter_dependencies() if d.flag.startswith('Line.Results.scaled_flow')]
        assert line_flows and all(d.params == {'scale': 2} for d in line_flows)

    def test_lines_inside_one_area_are_on_no_border(self, ds):
        for flag in ('CountryBorder.Results.flow_sent_up', 'MacroregionBorder.Results.flow_sent_up'):
            assert edges_on_no_border(ds, ds.flag_index.get_layer(flag)) == 2


class TestTradeBalanceOnABorderBase:
    FLAG = 'Country.Results.trade_balance_per_partner'

    def calculator_result(self, ds, losses):
        flows = NetworkLineFlowsData(
            sent_up=ds.fetch('Line.Results.flow_sent_up'),
            received_up=ds.fetch('Line.Results.flow_received_up'),
            sent_down=ds.fetch('Line.Results.flow_sent_down'),
            received_down=ds.fetch('Line.Results.flow_received_down'),
        )
        nodes = ds.fetch('Node.Model')
        calculator = RegionalTradeBalanceCalculator(ds.fetch('Line.Model'), nodes, 'country')
        return calculator.get_trade_balance(flows, losses)

    @pytest.mark.parametrize('losses', ['at_own_end', 'excluded'])
    def test_equals_calculator(self, ds, losses):
        result = ds.fetch(self.FLAG, losses=losses)
        expected = self.calculator_result(ds, losses)
        assert result.columns.names == ['country', 'partner_country', 'variable']
        pd.testing.assert_frame_equal(result.sort_index(axis=1), expected.sort_index(axis=1), check_names=False)

    def test_net_exports_with_losses_at_own_end_sum_to_the_losses(self, ds):
        net = ds.fetch(self.FLAG).xs('net_exp', level='variable', axis=1)
        losses = sum(
            ds.fetch(f'Line.Results.flow_sent_{d}')[['L1', 'L2', 'L4']].sum(axis=1)
            - ds.fetch(f'Line.Results.flow_received_{d}')[['L1', 'L2', 'L4']].sum(axis=1)
            for d in ('up', 'down')
        )
        pd.testing.assert_series_equal(net['C1', 'C2'] + net['C2', 'C1'], losses, check_names=False)

    def test_net_exports_with_losses_excluded_cancel(self, ds):
        net = ds.fetch(self.FLAG, losses='excluded').xs('net_exp', level='variable', axis=1)
        pd.testing.assert_series_equal(net['C1', 'C2'], -net['C2', 'C1'], check_names=False)

    def test_variables_split_net_export(self, ds):
        frame = ds.fetch(self.FLAG).sort_index(axis=1)['C1', 'C2']
        pd.testing.assert_series_equal(frame['exp'], frame['net_exp'].clip(0), check_names=False)
        pd.testing.assert_series_equal(frame['exp'] - frame['imp'], frame['net_exp'], check_names=False)
        pd.testing.assert_series_equal(frame['total'], frame['exp'] + frame['imp'], check_names=False)

    def test_total_sums_over_partners(self, ds):
        per_partner = ds.fetch(self.FLAG, losses='excluded')
        total = ds.fetch('Country.Results.trade_balance', losses='excluded')
        assert total.columns.names == ['country', 'variable']
        pd.testing.assert_series_equal(total['C1', 'net_exp'], per_partner['C1', 'C2', 'net_exp'], check_names=False)

    def test_losses_reach_aggregate_only(self, ds):
        ds.fetch(self.FLAG, losses='excluded')
        record = ds.get_fetch_record(self.FLAG, losses='excluded')
        flags = {d.flag: d.params for d in record.iter_dependencies()}
        assert record.params['losses'] == 'excluded'
        assert flags['CountryBorder.Results.flow_received_down'] == {}
        assert flags['Line.Results.flow_sent_up'] == {}

    def test_availability_reads_all_four_border_flows(self, ds):
        reason = ds.availability(self.FLAG).reason
        for side in ('sent', 'received'):
            for direction in ('up', 'down'):
                assert f'CountryBorder.Results.flow_{side}_{direction}' in reason

    def test_unknown_losses_raises(self, ds):
        with pytest.raises(ValueError, match='losses'):
            ds.fetch(self.FLAG, losses='half')

    def test_node_layer_without_border_model_has_no_trade_balance(self):
        scenario = DummyScenario(n_countries=1, n_nodes=2, n_lines=1, n_generators=4, n_demands=4)
        ds = LayerPlatform(scenario)
        assert ds.fetch(self.FLAG).columns.nlevels == 3
        assert ds.fetch(self.FLAG).empty


class TestParametersAndConfig:
    FLAG = 'Country.Results.demand_total'

    def test_base_params_reach_the_base_flag(self):
        ds = LayerPlatform()
        pd.testing.assert_frame_equal(ds.fetch(self.FLAG, solution='first'), ds.fetch(self.FLAG) * 0.98)

    def test_fetch_time_config_reaches_nested_fetches(self):
        ds = LayerPlatform()
        config = {'param_defaults': {'solution': 'first'}}
        pd.testing.assert_frame_equal(ds.fetch(self.FLAG, config=config), ds.fetch(self.FLAG, solution='first'))

    @pytest.mark.parametrize('flag', ['Country.Results.trade_balance_per_partner', 'Country.Results.trade_balance'])
    def test_every_nested_fetch_of_a_custom_aggregation_inherits_the_config_of_the_call(self, flag):
        ds = LayerPlatform()
        config = {'param_defaults': {'solution': 'first'}, 'price_offset': 100.0}
        ds.fetch(flag, config=config)
        nested = list(ds.get_fetch_record(flag, config=config).iter_dependencies())
        assert nested and all(record.call_config is INHERIT for record in nested)


class TestDescribe:
    AREA_MODELS = ['Country.Model', 'Macroregion.Model']
    BORDER_MODELS = ['CountryBorder.Model', 'MacroregionBorder.Model']

    def test_area_layers(self):
        layers = LayerPlatform.describe_class_area_layers()
        assert list(layers.index) == ['Country', 'Macroregion']
        assert layers.loc['Macroregion'].tolist() == [
            'macroregion',
            'Node.Model -> Macroregion.Model',
            'macroregion',
            5,
            'Custom regions of the nodes, for example the Nordic countries.',
        ]

    def test_border_layers(self):
        records = LayerPlatform.describe_class_border_layers(as_records=True)
        assert [r['name'] for r in records] == ['CountryBorder', 'MacroregionBorder']
        assert records[0]['refers_to'] == 'Country via Line.Model'
        assert records[0]['mapping_columns'] == ['node_from', 'node_to']
        assert records[0]['flags'] == 5

    @pytest.mark.parametrize(
        'spec, kind',
        [
            ('{area_layer}.Model', 'area'),
            ('{area_layer}.Results.price', 'area'),
            ('{area_layer}.Results.trade_balance', 'area'),
            ('{border_layer}.Model', 'border'),
            ('{border_layer}.Results.flow_{side}_{direction}', 'border'),
            ('{border_layer}.Results.price_spread', 'border'),
        ],
    )
    def test_layer_spec_names_its_layers_and_their_model_flags(self, ds, spec, kind):
        models = self.AREA_MODELS if kind == 'area' else self.BORDER_MODELS
        layers = [m.removesuffix('.Model') for m in models]
        for record in (
            LayerPlatform.describe_class_flags(as_records=True),
            ds.describe_flags(as_records=True),
        ):
            row = next(r for r in record if r['name'] == spec)
            assert (row['object'], row['layers']) == (models, layers)

    def test_object_matches_the_model_flag_of_each_flag(self, ds):
        spec = '{border_layer}.Constraints.capacity_{direction}'
        row = next(r for r in ds.describe_flags(as_records=True) if r['name'] == spec)
        model_flags = [ds.flag_info(f'{layer}.Constraints.capacity_up').model_flag for layer in row['layers']]
        assert row['object'] == model_flags
