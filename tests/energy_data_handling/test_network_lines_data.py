import pandas as pd
import pytest

from mesqual.energy_data_handling.network_lines_data import NetworkLineFlowsData


@pytest.fixture
def frame() -> pd.DataFrame:
    return pd.DataFrame({'line': [1.0, -2.0]}, index=pd.date_range('2024-01-01', periods=2, freq='h'))


def shifted_index(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.set_axis(frame.index + pd.Timedelta('1h'))


def renamed_column(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.rename(columns={'line': 'other'})


def test_flows_reject_granularity_argument(frame):
    with pytest.raises(TypeError):
        NetworkLineFlowsData(frame, frame, frame, frame, granularity=1.0)


@pytest.mark.parametrize('field', ['received_up', 'sent_down', 'received_down'])
def test_flows_index_mismatch_raises(frame, field):
    frames = {'sent_up': frame, 'received_up': frame, 'sent_down': frame, 'received_down': frame}
    frames[field] = shifted_index(frame)
    with pytest.raises(ValueError, match=f'Index mismatch: sent_up vs {field}'):
        NetworkLineFlowsData(**frames)


@pytest.mark.parametrize('field', ['received_up', 'sent_down', 'received_down'])
def test_flows_column_mismatch_raises(frame, field):
    frames = {'sent_up': frame, 'received_up': frame, 'sent_down': frame, 'received_down': frame}
    frames[field] = renamed_column(frame)
    with pytest.raises(ValueError, match=f'Columns mismatch: sent_up vs {field}'):
        NetworkLineFlowsData(**frames)


def test_net_flow_splits_sign_into_up_and_down(frame):
    flows = NetworkLineFlowsData.from_net_flow_without_losses(frame)
    assert flows.sent_up['line'].tolist() == [1.0, 0.0]
    assert flows.sent_down['line'].tolist() == [0.0, 2.0]
    assert flows.received_up.equals(flows.sent_up)
    assert flows.received_down.equals(flows.sent_down)


def test_equality_is_identity(frame):
    flows = NetworkLineFlowsData.from_net_flow_without_losses(frame)
    other = NetworkLineFlowsData.from_net_flow_without_losses(frame)
    assert flows == flows and flows != other
