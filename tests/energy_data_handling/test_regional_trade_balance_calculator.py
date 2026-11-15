import pandas as pd
import pytest

from mesqual.energy_data_handling.network_lines_data import NetworkLineFlowsData
from mesqual.energy_data_handling.variable_utils import RegionalTradeBalanceCalculator


@pytest.fixture
def calculator() -> RegionalTradeBalanceCalculator:
    line_model_df = pd.DataFrame({'node_from': ['A1', 'B2'], 'node_to': ['B1', 'A1']}, index=['L1', 'L2'])
    node_model_df = pd.DataFrame({'country': ['A', 'B', 'B']}, index=['A1', 'B1', 'B2'])
    return RegionalTradeBalanceCalculator(line_model_df, node_model_df, agg_region_column='country')


@pytest.fixture
def flow_data() -> NetworkLineFlowsData:
    index = pd.date_range('2024-01-01', periods=2, freq='h')

    def frame(l1: list[float], l2: list[float]) -> pd.DataFrame:
        return pd.DataFrame({'L1': l1, 'L2': l2}, index=index)

    return NetworkLineFlowsData(
        sent_up=frame([100, 0], [20, 0]),
        received_up=frame([98, 0], [19, 0]),
        sent_down=frame([0, 50], [0, 10]),
        received_down=frame([0, 49], [0, 9]),
    )


def _assert_values(trade_balance: pd.DataFrame, primary: str, partner: str, expected: dict[str, list[float]]):
    for variable, values in expected.items():
        assert trade_balance[(primary, partner, variable)].tolist() == values, variable


def test_at_own_end_is_the_default(calculator, flow_data):
    pd.testing.assert_frame_equal(
        calculator.get_trade_balance(flow_data),
        calculator.get_trade_balance(flow_data, losses='at_own_end'),
    )


def test_at_own_end_against_hand_computed_values(calculator, flow_data):
    trade_balance = calculator.get_trade_balance(flow_data, losses='at_own_end')
    _assert_values(trade_balance, 'A', 'B', {'net_exp': [81, -39], 'exp': [81, 0], 'imp': [0, 39], 'total': [81, 39]})
    _assert_values(trade_balance, 'B', 'A', {'net_exp': [-78, 41], 'exp': [0, 41], 'imp': [78, 0], 'total': [78, 41]})


def test_at_own_end_sides_sum_to_the_losses(calculator, flow_data):
    trade_balance = calculator.get_trade_balance(flow_data, losses='at_own_end')
    sides = trade_balance[('A', 'B', 'net_exp')] + trade_balance[('B', 'A', 'net_exp')]
    losses = (flow_data.sent_up - flow_data.received_up + flow_data.sent_down - flow_data.received_down).sum(axis=1)
    pd.testing.assert_series_equal(sides, losses, check_names=False)


def test_excluded_against_hand_computed_values(calculator, flow_data):
    trade_balance = calculator.get_trade_balance(flow_data, losses='excluded')
    _assert_values(trade_balance, 'A', 'B', {'net_exp': [80, -40], 'exp': [80, 0], 'imp': [0, 40], 'total': [80, 40]})
    _assert_values(trade_balance, 'B', 'A', {'net_exp': [-80, 40], 'exp': [0, 40], 'imp': [80, 0], 'total': [80, 40]})


def test_unknown_loss_mode_raises(calculator, flow_data):
    with pytest.raises(ValueError, match='losses'):
        calculator.get_trade_balance(flow_data, losses='post_loss')


@pytest.mark.parametrize('agg_region_column', ['country', None])
def test_line_end_node_without_row_in_the_node_model_raises(agg_region_column):
    line_model_df = pd.DataFrame(
        {'node_from': ['A1', 'X9', 'X9'], 'node_to': ['B1', 'A1', 'B1']}, index=['L1', 'L2', 'L3']
    )
    node_model_df = pd.DataFrame({'country': ['A', 'B']}, index=['A1', 'B1'])
    with pytest.raises(KeyError, match=r"nodes \['X9'\] have no row in the node model \(lines \['L2', 'L3'\]\)"):
        RegionalTradeBalanceCalculator(line_model_df, node_model_df, agg_region_column=agg_region_column)


def test_line_end_node_without_region_raises():
    line_model_df = pd.DataFrame({'node_from': ['A1', 'B2'], 'node_to': ['B1', 'A1']}, index=['L1', 'L2'])
    node_model_df = pd.DataFrame({'country': ['A', 'B', None]}, index=['A1', 'B1', 'B2'])
    with pytest.raises(
        KeyError, match=r"nodes \['B2'\] have no value in column 'country' of the node model \(lines \['L2'\]\)"
    ):
        RegionalTradeBalanceCalculator(line_model_df, node_model_df, agg_region_column='country')


def test_empty_line_end_node_raises():
    line_model_df = pd.DataFrame({'node_from': ['A1', None], 'node_to': ['B1', 'A1']}, index=[1, 2])
    node_model_df = pd.DataFrame({'country': ['A', 'B']}, index=['A1', 'B1'])
    with pytest.raises(
        KeyError, match=r"the line model has an empty end node in 'node_from' or 'node_to' \(lines \[2\]\)"
    ):
        RegionalTradeBalanceCalculator(line_model_df, node_model_df, agg_region_column='country')


def test_all_three_faults_in_one_message():
    line_model_df = pd.DataFrame(
        {'node_from': ['A1', None, 'X9', 'B2'], 'node_to': ['B1', 'A1', 'A1', 'A1']}, index=['L1', 'L2', 'L3', 'L4']
    )
    node_model_df = pd.DataFrame({'country': ['A', 'B', None]}, index=['A1', 'B1', 'B2'])
    with pytest.raises(
        KeyError,
        match=(
            r"Invalid line end nodes: the line model has an empty end node in 'node_from' or 'node_to' "
            r"\(lines \['L2'\]\); nodes \['X9'\] have no row in the node model \(lines \['L3'\]\); "
            r"nodes \['B2'\] have no value in column 'country' of the node model \(lines \['L4'\]\)\."
        ),
    ):
        RegionalTradeBalanceCalculator(line_model_df, node_model_df, agg_region_column='country')
