import pandas as pd
import pytest

from mesqual.energy_data_handling.granularity_analyzer import GranularityError, TimeSeriesGranularityAnalyzer


def test_empty_index_gives_empty_series():
    analyzer = TimeSeriesGranularityAnalyzer()
    index = pd.DatetimeIndex([], tz='UTC')
    timedeltas = analyzer.get_granularity_as_series_of_timedeltas(index)
    hours = analyzer.get_granularity_as_series_of_hours(index)
    assert timedeltas.empty and timedeltas.index.equals(index)
    assert hours.empty and hours.dtype == float


def test_unsorted_index_gives_the_steps_in_input_order():
    index = pd.date_range('2024-01-01', periods=24, freq='h', name='time').append(
        pd.date_range('2024-01-02', periods=96, freq='15min', name='time')
    )
    shuffled = index[pd.Series(range(len(index))).sample(frac=1, random_state=0).to_numpy()]
    hours = TimeSeriesGranularityAnalyzer().get_granularity_as_series_of_hours(shuffled)
    assert not shuffled.is_monotonic_increasing
    assert hours.index.equals(shuffled)
    expected = pd.Series([1.0] * 24 + [0.25] * 96, index=index)
    pd.testing.assert_series_equal(hours, expected.loc[shuffled])


def test_duplicate_timestamps_raise_in_strict_mode():
    index = pd.date_range('2024-01-01', periods=24, freq='h')
    with pytest.raises(GranularityError, match='duplicate'):
        TimeSeriesGranularityAnalyzer().get_granularity_as_series_of_hours(index.append(index[:1]))


def test_later_occurrence_of_a_timestamp_gets_zero_without_strict_mode():
    index = pd.date_range('2024-01-01', periods=24, freq='h')
    with_duplicate = index[:6].append(index[5:])
    with pytest.warns(UserWarning, match='duplicate'):
        hours = TimeSeriesGranularityAnalyzer(strict_mode=False).get_granularity_as_series_of_hours(with_duplicate)
    assert hours.iloc[5] == 1.0 and hours.iloc[6] == 0.0
    assert hours.sum() == 24.0


def test_nat_in_index_raises():
    index = pd.DatetimeIndex(['2024-01-01 00:00', None, '2024-01-01 02:00'])
    with pytest.raises(ValueError, match='NaT'):
        TimeSeriesGranularityAnalyzer(strict_mode=False).get_granularity_as_series_of_hours(index)


def test_day_with_one_row():
    index = pd.date_range('2024-01-01', periods=24, freq='h').append(pd.DatetimeIndex(['2024-01-02 00:00']))
    with pytest.raises(GranularityError, match='one row'):
        TimeSeriesGranularityAnalyzer().get_granularity_as_series_of_hours(index)
    hours = TimeSeriesGranularityAnalyzer(strict_mode=False).get_granularity_as_series_of_hours(index)
    assert hours.iloc[:24].eq(1.0).all() and pd.isna(hours.iloc[24])


def test_repeat_of_the_last_row_of_a_day_counts_once():
    index = pd.date_range('2024-01-01', periods=24, freq='h')
    with_repeat = index.append(index[-1:])
    with pytest.warns(UserWarning, match='duplicate'):
        hours = TimeSeriesGranularityAnalyzer(strict_mode=False).get_granularity_as_series_of_hours(with_repeat)
    assert hours.iloc[23] == 1.0 and hours.iloc[24] == 0.0
    assert hours.sum() == 24.0


def test_reference_step_is_the_step_of_the_earliest_timestamp():
    index = pd.date_range('2024-01-01', periods=24, freq='h').append(
        pd.date_range('2024-01-02', periods=96, freq='15min')
    )
    analyzer = TimeSeriesGranularityAnalyzer(strict_mode=False)
    with pytest.warns(UserWarning, match='Multiple granularities'):
        reversed_step = analyzer.get_granularity_as_timedelta(index[::-1])
    with pytest.warns(UserWarning, match='Multiple granularities'):
        step = analyzer.get_granularity_as_timedelta(index)
    assert reversed_step == pd.Timedelta('1h')
    assert step == pd.Timedelta('1h')


def test_duplicates_give_no_warning_of_multiple_granularities():
    index = pd.date_range('2024-01-01', periods=24, freq='h')
    with pytest.warns(UserWarning) as record:
        step = TimeSeriesGranularityAnalyzer(strict_mode=False).get_granularity_as_timedelta(index.append(index[:1]))
    assert step == pd.Timedelta('1h')
    assert not [w for w in record if 'Multiple granularities' in str(w.message)]


def test_reference_step_of_nat_warns_without_strict_mode():
    index = pd.DatetimeIndex(['2024-01-01 00:00']).append(pd.date_range('2024-01-02', periods=24, freq='h'))
    with pytest.warns(UserWarning, match='no step length'):
        step = TimeSeriesGranularityAnalyzer(strict_mode=False).get_granularity_as_timedelta(index)
    assert pd.isna(step)
