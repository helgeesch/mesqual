import pandas as pd

from mesqual.energy_data_handling.network_lines_data import NetworkLineFlowsData
from mesqual.energy_data_handling.variable_utils import RegionalTradeBalanceCalculator, trade_balance_frame

INDEX = pd.RangeIndex(2, name='time')


class TestTradeBalanceFrame:
    def test_splits_net_export_into_variables(self):
        net = pd.Series([100.0, -40.0], index=INDEX)
        frame = trade_balance_frame({('A', 'B'): net, ('B', 'A'): -net}, 'area', 'partner_area')
        assert frame.columns.names == ['area', 'partner_area', 'variable']
        assert frame['A', 'B', 'exp'].tolist() == [100.0, 0.0]
        assert frame['A', 'B', 'imp'].tolist() == [0.0, 40.0]
        assert frame['A', 'B', 'total'].tolist() == [100.0, 40.0]
        assert frame['B', 'A', 'net_exp'].tolist() == [-100.0, 40.0]

    def test_empty_input_gives_the_index_and_the_levels(self):
        frame = trade_balance_frame({}, 'area', 'partner_area', INDEX)
        assert frame.empty and frame.index.equals(INDEX)
        assert frame.columns.names == ['area', 'partner_area', 'variable']


class TestCalculator:
    def test_results_per_loss_mode(self):
        lines = pd.DataFrame({'node_from': ['a1', 'b1'], 'node_to': ['b1', 'a2']}, index=['l1', 'l2'])
        nodes = pd.DataFrame({'country': ['A', 'A', 'B']}, index=['a1', 'a2', 'b1'])
        up = pd.DataFrame({'l1': [100.0, 0.0], 'l2': [0.0, 10.0]}, index=INDEX)
        down = pd.DataFrame({'l1': [0.0, 50.0], 'l2': [20.0, 0.0]}, index=INDEX)
        flows = NetworkLineFlowsData(sent_up=up, received_up=up * 0.9, sent_down=down, received_down=down * 0.9)
        calculator = RegionalTradeBalanceCalculator(lines, nodes, 'country')
        post = calculator.get_trade_balance(flows, losses='at_own_end')
        pre = calculator.get_trade_balance(flows, losses='excluded')
        assert post['A', 'B', 'net_exp'].tolist() == [100.0 + 20.0, -45.0 - 9.0]
        assert pre['A', 'B', 'net_exp'].tolist() == [120.0, -60.0]
        assert pre['B', 'A', 'net_exp'].tolist() == [-120.0, 60.0]
        assert post.columns.names == ['primary_country', 'partner_country', 'variable']
