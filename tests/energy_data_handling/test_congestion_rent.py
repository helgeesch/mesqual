import pandas as pd
import pytest

from mesqual.energy_data_handling.granularity_analyzer import GranularityError
from mesqual.energy_data_handling.network_lines_data import NetworkLineFlowsData
from mesqual.energy_data_handling.variable_utils import CongestionRentCalculator


@pytest.fixture
def index() -> pd.DatetimeIndex:
    """One day of 60 min steps, then one day of 15 min steps."""
    hourly = pd.date_range('2024-01-01', periods=24, freq='h', tz='Europe/Berlin')
    quarter_hourly = pd.date_range('2024-01-02', periods=96, freq='15min', tz='Europe/Berlin')
    return hourly.append(quarter_hourly)


@pytest.fixture
def flows(index) -> NetworkLineFlowsData:
    n = len(index)
    sent_up = pd.DataFrame({'L1': [100.0] * n, 'L2': [0.0] * n}, index=index)
    sent_down = pd.DataFrame({'L1': [0.0] * n, 'L2': [40.0] * n}, index=index)
    return NetworkLineFlowsData(
        sent_up=sent_up, received_up=sent_up * 0.9, sent_down=sent_down, received_down=sent_down * 0.5
    )


@pytest.fixture
def line_model() -> pd.DataFrame:
    return pd.DataFrame({'node_from': ['A', 'B'], 'node_to': ['B', 'C']}, index=['L1', 'L2'])


@pytest.fixture
def node_prices(index) -> pd.DataFrame:
    n = len(index)
    return pd.DataFrame({'A': [10.0] * n, 'B': [50.0] * n, 'C': [20.0] * n}, index=index)


@pytest.fixture
def calculator(flows, line_model, node_prices) -> CongestionRentCalculator:
    return CongestionRentCalculator.from_node_prices(flows, line_model, node_prices)


def test_frame_input_uses_the_step_length_per_day(calculator):
    rent = calculator.calculate()
    hourly, quarter_hourly = rent.iloc[:24], rent.iloc[24:]
    assert (hourly['L1'] == 90 * 50 - 100 * 10).all()
    assert (quarter_hourly['L1'] == (90 * 50 - 100 * 10) * 0.25).all()
    assert (hourly['L2'] == 20 * 50 - 40 * 20).all()
    assert (quarter_hourly['L2'] == (20 * 50 - 40 * 20) * 0.25).all()
    assert rent['L1'].sum() == pytest.approx(2 * 24 * 3500)


def test_frame_input_equals_series_input_per_line(calculator):
    rent = calculator.calculate()
    for line in rent.columns:
        per_line = CongestionRentCalculator(
            sent_up=calculator.sent_up[line],
            received_up=calculator.received_up[line],
            sent_down=calculator.sent_down[line],
            received_down=calculator.received_down[line],
            price_node_from=calculator.price_node_from[line],
            price_node_to=calculator.price_node_to[line],
        )
        pd.testing.assert_series_equal(rent[line], per_line.calculate())


def test_granularity_series_applies_along_rows(flows, line_model, node_prices, index):
    granularity = pd.Series(2.0, index=index)
    rent = CongestionRentCalculator.from_node_prices(flows, line_model, node_prices, granularity_hrs=granularity)
    assert (rent.calculate()['L1'] == 2 * 3500).all()


def test_from_node_prices_maps_prices_to_line_ends(calculator, node_prices):
    pd.testing.assert_series_equal(calculator.price_node_from['L2'], node_prices['B'].rename('L2'))
    pd.testing.assert_series_equal(calculator.price_node_to['L2'], node_prices['C'].rename('L2'))


def test_from_node_prices_with_other_column_names(calculator, flows, line_model, node_prices):
    renamed = line_model.rename(columns={'node_from': 'a', 'node_to': 'b'})
    other = CongestionRentCalculator.from_node_prices(flows, renamed, node_prices, node_from_col='a', node_to_col='b')
    pd.testing.assert_frame_equal(other.calculate(), calculator.calculate())


def test_from_node_prices_raises_for_missing_line_or_node(flows, line_model, node_prices):
    with pytest.raises(KeyError, match='L2'):
        CongestionRentCalculator.from_node_prices(flows, line_model.loc[['L1']], node_prices)
    with pytest.raises(KeyError, match='C'):
        CongestionRentCalculator.from_node_prices(flows, line_model, node_prices.drop(columns='C'))


def test_auto_detection_raises_for_mixed_step_lengths_within_a_day():
    index = pd.date_range('2024-01-01', periods=4, freq='15min').append(
        pd.date_range('2024-01-01 01:00', periods=23, freq='h')
    )
    values = pd.Series(1.0, index=index)
    with pytest.raises(GranularityError):
        CongestionRentCalculator.from_net_flow_without_losses(values, values, values)


def test_auto_detection_raises_for_a_missing_row_within_a_day():
    index = pd.date_range('2024-01-01', periods=24, freq='h').delete(5)
    values = pd.Series(1.0, index=index)
    with pytest.raises(GranularityError, match='Pass granularity_hrs'):
        CongestionRentCalculator.from_net_flow_without_losses(values, values, values)


def test_explicit_granularity_is_trusted():
    index = pd.date_range('2024-01-01', periods=24, freq='h').delete(5)
    values = pd.Series(1.0, index=index)
    rent = CongestionRentCalculator.from_net_flow_without_losses(values, values, values * 3, granularity_hrs=1)
    assert (rent == 2.0).all()


def test_mixed_series_and_frame_inputs_raise(calculator):
    with pytest.raises(TypeError):
        CongestionRentCalculator(
            sent_up=calculator.sent_up,
            received_up=calculator.received_up,
            sent_down=calculator.sent_down,
            received_down=calculator.received_down,
            price_node_from=calculator.price_node_from['L1'],
            price_node_to=calculator.price_node_to,
        )


def test_column_mismatch_raises(calculator):
    with pytest.raises(ValueError, match='Columns mismatch'):
        CongestionRentCalculator(
            sent_up=calculator.sent_up,
            received_up=calculator.received_up,
            sent_down=calculator.sent_down,
            received_down=calculator.received_down,
            price_node_from=calculator.price_node_from[['L2', 'L1']],
            price_node_to=calculator.price_node_to,
        )


def test_empty_index_gives_empty_rent():
    values = pd.Series([], index=pd.DatetimeIndex([]), dtype=float)
    assert CongestionRentCalculator.from_net_flow_without_losses(values, values, values).empty


def test_auto_detection_raises_for_a_day_with_one_row():
    index = pd.date_range('2024-01-01', periods=24, freq='h').append(pd.DatetimeIndex(['2024-01-02 00:00']))
    values = pd.Series(1.0, index=index)
    with pytest.raises(GranularityError, match='Pass granularity_hrs'):
        CongestionRentCalculator.from_net_flow_without_losses(values, values, values)


def test_granularity_of_other_type_raises():
    values = pd.Series(1.0, index=pd.date_range('2024-01-01', periods=3, freq='h'))
    with pytest.raises(TypeError, match='Series or a number'):
        CongestionRentCalculator.from_net_flow_without_losses(values, values, values, granularity_hrs='1h')
    with pytest.raises(TypeError, match='Series or a number'):
        CongestionRentCalculator.from_net_flow_without_losses(values, values, values, granularity_hrs=True)


def test_granularity_series_with_another_index_raises():
    index = pd.date_range('2024-01-01', periods=3, freq='h')
    values = pd.Series(1.0, index=index)
    with pytest.raises(ValueError, match='index of granularity_hrs'):
        CongestionRentCalculator.from_net_flow_without_losses(
            values, values, values, granularity_hrs=pd.Series(1.0, index=index.shift(1))
        )


@pytest.mark.parametrize('day, n_hours', [('2024-03-31', 23), ('2024-10-27', 25)])
def test_auto_detection_on_a_dst_day(day, n_hours):
    index = pd.date_range(day, pd.Timestamp(day) + pd.Timedelta(days=1), freq='h', tz='Europe/Berlin', inclusive='left')
    values = pd.Series(1.0, index=index)
    rent = CongestionRentCalculator.from_net_flow_without_losses(values, values, values * 3)
    assert len(index) == n_hours
    assert (rent == 2.0).all()


@pytest.mark.parametrize('granularity', [pd.Series(['1h'] * 3), pd.Series([True] * 3)])
def test_granularity_series_of_other_dtype_raises(granularity):
    index = pd.date_range('2024-01-01', periods=3, freq='h')
    values = pd.Series(1.0, index=index)
    with pytest.raises(TypeError, match='numeric'):
        CongestionRentCalculator.from_net_flow_without_losses(
            values, values, values, granularity_hrs=granularity.set_axis(index)
        )
