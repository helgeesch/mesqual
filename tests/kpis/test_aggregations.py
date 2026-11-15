import pandas as pd
import pytest

from mesqual.kpis.aggregations import Aggregations


@pytest.fixture
def frame() -> pd.DataFrame:
    index = pd.date_range('2024-01-01', periods=24, freq='h').append(
        pd.date_range('2024-01-02', periods=96, freq='15min')
    )
    return pd.DataFrame({'a': range(len(index))}, index=index, dtype=float)


@pytest.mark.parametrize('aggregation', [Aggregations.AnnualizedSum, Aggregations.DailySum])
def test_unsorted_index_gives_the_sorted_result(frame, aggregation):
    shuffled = frame.sample(frac=1, random_state=0)
    pd.testing.assert_series_equal(aggregation(shuffled), aggregation(frame))


def test_duplicate_timestamp_counts_each_timestamp_once():
    index = pd.date_range('2024-01-01', periods=24, freq='h')
    frame = pd.DataFrame({'a': 1.0}, index=index[:6].append(index[5:]))
    with pytest.warns(UserWarning, match='duplicate timestamps'):
        annualized = Aggregations.AnnualizedSum(frame)['a']
    with pytest.warns(UserWarning, match='duplicate timestamps'):
        daily = Aggregations.DailySum(frame)['a']
    assert annualized == pytest.approx(25 / 24 * 8760)
    assert daily == pytest.approx(25.0)


def test_repeat_of_the_last_row_of_a_day_counts_once():
    index = pd.date_range('2024-01-01', periods=24, freq='h')
    frame = pd.DataFrame({'a': 1.0}, index=index.append(index[-1:]))
    with pytest.warns(UserWarning, match='duplicate timestamps'):
        daily = Aggregations.DailySum(frame)['a']
    assert daily == pytest.approx(25.0)
