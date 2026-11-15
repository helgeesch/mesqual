import pandas as pd
import pytest

from mesqual.energy_data_handling import TimeSeriesGapHandler

INDEX = pd.date_range('2024-01-01', periods=5, freq='h').delete([2, 3])
"""00:00, 01:00 and 04:00: a gap of three hours after 01:00."""


def as_kind(kind: str) -> pd.Series | pd.DataFrame:
    series = pd.Series([1.0, 2.0, 3.0], index=INDEX, name='value')
    return series if kind == 'series' else series.to_frame().assign(other=[4.0, 5.0, 6.0])


@pytest.mark.parametrize('kind', ['series', 'frame'])
@pytest.mark.parametrize('order', [[0, 1, 2], [2, 1, 0], [0, 2, 1]], ids=['sorted', 'reversed', 'partly sorted'])
def test_gap_longer_than_the_maximum_gets_one_nan_row_at_the_maximum_after_its_last_row(kind, order):
    data = as_kind(kind).iloc[order]
    result = TimeSeriesGapHandler(max_gap_in_minutes=90).insert_nans_at_gaps(data)
    expected = as_kind(kind).reindex(INDEX.insert(2, pd.Timestamp('2024-01-01 02:30')))
    assert result.equals(expected)


@pytest.mark.parametrize('order', ['sorted', 'reversed'])
def test_gaps_up_to_the_maximum_give_no_row_and_a_sorted_result(order):
    data = as_kind('series') if order == 'sorted' else as_kind('series').iloc[::-1]
    result = TimeSeriesGapHandler(max_gap_in_minutes=180).insert_nans_at_gaps(data)
    pd.testing.assert_series_equal(result, as_kind('series'))


def test_index_that_is_not_a_datetime_index_raises():
    with pytest.raises(TypeError, match='DatetimeIndex'):
        TimeSeriesGapHandler().insert_nans_at_gaps(pd.Series([1.0, 2.0], index=[0, 1]))
