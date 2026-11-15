import numpy as np
import pandas as pd
import pytest

from mesqual.energy_data_handling.granularity_converter import TimeSeriesGranularityConverter
from mesqual.enums import QuantityTypeEnum
from mesqual.units import Units


def assert_index_equal_ignoring_freq(result: pd.DatetimeIndex, expected: pd.DatetimeIndex) -> None:
    pd.testing.assert_index_equal(pd.DatetimeIndex(result, freq=None), pd.DatetimeIndex(expected, freq=None))


@pytest.fixture
def hourly() -> pd.Series:
    return pd.Series(1.0, index=pd.date_range('2024-01-01', periods=24, freq='h'))


@pytest.mark.parametrize('quantity_type', [QuantityTypeEnum.UNKNOWN, QuantityTypeEnum.AMBIGUOUS])
def test_unknown_and_ambiguous_quantity_types_raise(hourly, quantity_type):
    with pytest.raises(ValueError, match='quantity type'):
        TimeSeriesGranularityConverter().convert(hourly, '15min', quantity_type)


def test_extensive_upsampling_splits_the_value(hourly):
    result = TimeSeriesGranularityConverter().convert(hourly, '15min', QuantityTypeEnum.EXTENSIVE)
    assert result.iloc[0] == pytest.approx(0.25)


@pytest.mark.parametrize('unit, daily_value', [(Units.EUR_per_MW_per_period, 24.0), (Units.EUR_per_MW, 1.0)])
def test_capacity_price_per_period_sums_and_per_hour_averages(hourly, unit, daily_value):
    quantity_type = Units.get_quantity_type_enum(unit)
    result = TimeSeriesGranularityConverter().convert(hourly, '1D', quantity_type)
    assert result.iloc[0] == pytest.approx(daily_value)


def as_kind(series: pd.Series, kind: str) -> pd.Series | pd.DataFrame:
    return series if kind == 'series' else pd.DataFrame({'a': series, 'b': series * 2})


@pytest.mark.parametrize('kind', ['series', 'frame'])
class TestUpsamplingCoversEveryStep:
    """Upsampling covers the last input step too."""

    @pytest.mark.parametrize('start, periods', [('2024-01-01', 24), ('2025-03-30', 23), ('2025-10-26', 25)])
    def test_n_hourly_values_give_4n_quarter_hours(self, kind, start, periods):
        hourly = as_kind(
            pd.Series(1.0, index=pd.date_range(start, periods=periods, freq='h', tz='Europe/Berlin')), kind
        )
        result = TimeSeriesGranularityConverter().convert(hourly, '15min', QuantityTypeEnum.INTENSIVE)
        assert len(result) == 4 * periods
        assert (result.sum() == 4 * hourly.sum()).all()

    def test_extensive_splits_every_value(self, kind):
        hourly = as_kind(pd.Series([1.0, 2.0, 3.0], index=pd.date_range('2024-01-01', periods=3, freq='h')), kind)
        result = TimeSeriesGranularityConverter().convert(hourly, '15min', QuantityTypeEnum.EXTENSIVE)
        expected = as_kind(
            pd.Series([0.25] * 4 + [0.5] * 4 + [0.75] * 4, index=pd.date_range('2024-01-01', periods=12, freq='15min')),
            kind,
        )
        assert_index_equal_ignoring_freq(result.index, expected.index)
        assert result.to_numpy().tolist() == expected.to_numpy().tolist()

    def test_a_step_change_at_a_day_boundary_expands_each_day_by_its_own_step(self, kind):
        hourly_index = pd.date_range('2024-01-01', periods=24, freq='h')
        quarter_index = pd.date_range('2024-01-02', periods=96, freq='15min')
        mixed = as_kind(pd.concat([pd.Series(4.0, index=hourly_index), pd.Series(1.0, index=quarter_index)]), kind)
        result = TimeSeriesGranularityConverter().convert(mixed, '15min', QuantityTypeEnum.EXTENSIVE)
        expected_index = pd.date_range(hourly_index[0], quarter_index[-1], freq='15min')
        assert_index_equal_ignoring_freq(result.index, expected_index)
        assert (result.sum() == mixed.sum()).all()
        assert (result == as_kind(pd.Series(1.0, index=expected_index), kind)).all(axis=None)

    def test_a_day_with_one_row_takes_the_step_of_the_day_before(self, kind):
        hourly = as_kind(pd.Series(1.0, index=pd.date_range('2025-01-01', '2025-01-02', freq='h')), kind)
        result = TimeSeriesGranularityConverter().convert(hourly, '15min', QuantityTypeEnum.EXTENSIVE)
        assert len(result) == 100
        assert result.index[-1] == pd.Timestamp('2025-01-02 00:45')
        assert (result.sum() == hourly.sum()).all()
        assert (result == result.iloc[0]).all(axis=None)

    def test_a_first_day_with_one_row_takes_the_step_of_the_day_after(self, kind):
        index = pd.DatetimeIndex(['2024-12-31 23:00']).append(pd.date_range('2025-01-01', periods=24, freq='h'))
        result = TimeSeriesGranularityConverter().convert(
            as_kind(pd.Series(1.0, index=index), kind), '15min', QuantityTypeEnum.INTENSIVE
        )
        assert len(result) == 100

    def test_a_series_of_one_row_is_unchanged(self, kind):
        single = as_kind(pd.Series(2.0, index=pd.DatetimeIndex(['2025-01-01'])), kind)
        result = TimeSeriesGranularityConverter().convert(single, '15min', QuantityTypeEnum.EXTENSIVE)
        assert result.to_numpy().tolist() == single.to_numpy().tolist()
        pd.testing.assert_index_equal(result.index, single.index)

    @pytest.mark.parametrize('quantity_type', [QuantityTypeEnum.INTENSIVE, QuantityTypeEnum.EXTENSIVE])
    def test_a_quarter_hourly_day_with_a_gap_is_unchanged(self, kind, quantity_type):
        index = pd.date_range('2024-01-01', periods=96, freq='15min')
        data = as_kind(pd.Series(1.0, index=index[(index.hour != 10)]), kind)
        result = TimeSeriesGranularityConverter().convert(data, '15min', quantity_type)
        pd.testing.assert_index_equal(result.index, data.index)
        assert result.to_numpy().tolist() == data.to_numpy().tolist()


@pytest.mark.parametrize('kind', ['series', 'frame'])
class TestGapsStayGaps:
    """Upsampling repeats each row by its native step and never fills a gap."""

    def test_rows_before_and_after_a_gap_cover_their_own_step(self, kind):
        index = pd.DatetimeIndex(['2024-01-01 00:00', '2024-01-01 01:00', '2024-01-01 05:00'])
        data = as_kind(pd.Series([4.0, 8.0, 12.0], index=index), kind)
        result = TimeSeriesGranularityConverter().convert(data, '15min', QuantityTypeEnum.EXTENSIVE)
        expected_index = pd.date_range('2024-01-01 00:00', periods=8, freq='15min').append(
            pd.date_range('2024-01-01 05:00', periods=4, freq='15min')
        )
        assert_index_equal_ignoring_freq(result.index, expected_index)
        expected = as_kind(pd.Series([1.0] * 4 + [2.0] * 4 + [3.0] * 4, index=expected_index), kind)
        assert result.to_numpy().tolist() == expected.to_numpy().tolist()

    def test_a_missing_hour_gives_no_quarter_hours(self, kind):
        index = pd.date_range('2024-01-01', periods=24, freq='h')
        data = as_kind(pd.Series(4.0, index=index[index.hour != 10]), kind)
        result = TimeSeriesGranularityConverter().convert(data, '15min', QuantityTypeEnum.EXTENSIVE)
        assert len(result) == 92
        assert 10 not in result.index.hour
        before_gap = result.loc['2024-01-01 09:00':'2024-01-01 09:45']
        assert len(before_gap) == 4
        assert np.all(before_gap.sum() == data.iloc[9])

    def test_a_missing_day_gives_no_rows(self, kind):
        index = pd.date_range('2024-01-01', periods=72, freq='h')
        data = as_kind(pd.Series(1.0, index=index[index.day != 2]), kind)
        result = TimeSeriesGranularityConverter().convert(data, '15min', QuantityTypeEnum.INTENSIVE)
        assert len(result) == 4 * 48
        assert 2 not in result.index.day

    @pytest.mark.parametrize('quantity_type', [QuantityTypeEnum.INTENSIVE, QuantityTypeEnum.EXTENSIVE])
    def test_downsampling_gives_no_row_for_a_target_step_without_input(self, kind, quantity_type):
        index = pd.date_range('2024-01-01', periods=96, freq='15min')
        data = as_kind(pd.Series(1.0, index=index[index.hour != 10]), kind)
        result = TimeSeriesGranularityConverter().convert(data, '1h', quantity_type)
        assert len(result) == 23
        assert 10 not in result.index.hour


class TestNaNIsAValue:
    """A gap is a missing timestamp; a NaN value is a value of its row."""

    @pytest.mark.parametrize('quantity_type', [QuantityTypeEnum.INTENSIVE, QuantityTypeEnum.EXTENSIVE])
    def test_a_series_with_nan_at_its_own_step_is_unchanged(self, quantity_type):
        hourly = pd.Series([1.0, float('nan'), 3.0], index=pd.date_range('2024-01-01', periods=3, freq='h'))
        result = TimeSeriesGranularityConverter().convert(hourly, '1h', quantity_type)
        pd.testing.assert_series_equal(result, hourly, check_freq=False)

    def test_a_sparse_column_upsamples_without_new_rows_and_nan_stays_nan(self):
        index = pd.date_range('2024-01-01', periods=24, freq='h')
        data = pd.DataFrame({'full': 4.0, 'sparse': float('nan')}, index=index)
        data.loc[index[::6], 'sparse'] = 8.0
        result = TimeSeriesGranularityConverter().convert(data, '15min', QuantityTypeEnum.EXTENSIVE)
        assert len(result) == 96
        assert result['sparse'].notna().sum() == 16
        assert result['sparse'].sum() == data['sparse'].sum()
        assert (result['full'] == 1.0).all()

    def test_a_sparse_column_at_its_own_step_is_unchanged(self):
        index = pd.date_range('2024-01-01', periods=24, freq='h')
        data = pd.DataFrame({'full': 4.0, 'sparse': float('nan')}, index=index)
        data.loc[index[::6], 'sparse'] = 8.0
        result = TimeSeriesGranularityConverter().convert(data, '1h', QuantityTypeEnum.INTENSIVE)
        pd.testing.assert_frame_equal(result, data, check_freq=False)

    def test_an_hourly_column_on_a_quarter_hourly_index_keeps_its_nan_rows(self):
        hourly = pd.Series(4.0, index=pd.date_range('2024-01-01', periods=24, freq='h'))
        quarter_hourly = pd.Series(1.0, index=pd.date_range('2024-01-01', periods=96, freq='15min'))
        data = pd.DataFrame({'h': hourly, 'q': quarter_hourly})
        result = TimeSeriesGranularityConverter().convert(data, '15min', QuantityTypeEnum.EXTENSIVE)
        pd.testing.assert_frame_equal(result, data, check_freq=False)

    @pytest.mark.parametrize('quantity_type', [QuantityTypeEnum.INTENSIVE, QuantityTypeEnum.EXTENSIVE])
    def test_a_target_step_with_only_nan_rows_gives_nan(self, quantity_type):
        index = pd.date_range('2024-01-01', periods=8, freq='15min')
        data = pd.Series([float('nan')] * 4 + [1.0] * 4, index=index)
        result = TimeSeriesGranularityConverter().convert(data, '1h', quantity_type)
        assert len(result) == 2
        assert np.isnan(result.iloc[0])


class TestInputChecks:
    def test_unsorted_input_is_sorted_first(self):
        hourly = pd.Series([1.0, 2.0, 3.0], index=pd.date_range('2024-01-01', periods=3, freq='h'))
        result = TimeSeriesGranularityConverter().convert(hourly.iloc[::-1], '15min', QuantityTypeEnum.INTENSIVE)
        expected = TimeSeriesGranularityConverter().convert(hourly, '15min', QuantityTypeEnum.INTENSIVE)
        pd.testing.assert_series_equal(result, expected)

    def test_duplicate_timestamps_raise(self):
        index = pd.DatetimeIndex(['2024-01-01 00:00', '2024-01-01 00:00', '2024-01-01 01:00'])
        with pytest.raises(ValueError, match='duplicates'):
            TimeSeriesGranularityConverter().convert(pd.Series(1.0, index=index), '15min', QuantityTypeEnum.INTENSIVE)

    def test_duplicate_timestamps_in_one_group_raise(self):
        index = pd.MultiIndex.from_tuples(
            [('a', pd.Timestamp('2024-01-01')), ('a', pd.Timestamp('2024-01-01')), ('b', pd.Timestamp('2024-01-01'))],
            names=['dataset', 'datetime'],
        )
        with pytest.raises(ValueError, match='duplicates'):
            TimeSeriesGranularityConverter().convert(
                pd.Series(1.0, index=index), '15min', QuantityTypeEnum.INTENSIVE, time_level='datetime'
            )

    def test_a_group_of_one_row_takes_the_step_of_the_whole_input(self):
        index = pd.date_range('2024-01-01', periods=3, freq='h', name='datetime')
        data = pd.concat({'a': pd.Series(4.0, index=index), 'b': pd.Series(4.0, index=index[:1])}, names=['dataset'])
        result = TimeSeriesGranularityConverter().convert(
            data, '15min', QuantityTypeEnum.EXTENSIVE, time_level='datetime'
        )
        assert len(result.loc['b']) == 4
        assert (result == 1.0).all()

    @pytest.mark.parametrize('periods', [0, 1])
    def test_a_target_index_without_a_step_raises(self, hourly, periods):
        with pytest.raises(ValueError, match='at least two rows'):
            TimeSeriesGranularityConverter().convert_to_target_index(
                hourly, pd.date_range('2024-01-01', periods=periods, freq='15min'), QuantityTypeEnum.INTENSIVE
            )


@pytest.mark.parametrize('start, hours', [('2025-03-29', [24, 23, 24]), ('2025-10-25', [24, 25, 24])])
@pytest.mark.parametrize('quantity_type', [QuantityTypeEnum.INTENSIVE, QuantityTypeEnum.EXTENSIVE])
def test_daily_to_hourly_on_dst_days_gives_calendar_days(start, hours, quantity_type):
    values = [float(h) for h in hours] if quantity_type == QuantityTypeEnum.EXTENSIVE else [1.0] * 3
    daily = pd.Series(values, index=pd.date_range(start, periods=3, freq='D', tz='Europe/Berlin'))
    result = TimeSeriesGranularityConverter().convert(daily, '1h', quantity_type)
    assert result.groupby(result.index.date).size().tolist() == hours
    assert result.index.is_unique
    assert (result == 1.0).all()


@pytest.mark.parametrize('empty', [pd.Series(dtype=float), pd.DataFrame(), pd.DataFrame(index=pd.Index([], name='x'))])
def test_empty_input_gives_a_copy_for_any_index(empty):
    result = TimeSeriesGranularityConverter().convert(empty, '1h', QuantityTypeEnum.INTENSIVE)
    assert result.empty
    assert result is not empty


@pytest.mark.parametrize('kind', ['series', 'frame'])
class TestOneRowPerDay:
    """A day with one row takes a step from other days or from the whole series."""

    def test_daily_to_hourly_extensive_splits_each_day(self, kind):
        daily = as_kind(pd.Series([24.0, 48.0, 72.0], index=pd.date_range('2024-01-01', periods=3, freq='D')), kind)
        result = TimeSeriesGranularityConverter().convert(daily, '1h', QuantityTypeEnum.EXTENSIVE)
        assert len(result) == 72
        assert (result.sum() == daily.sum()).all()
        assert result.iloc[[0, 30, 60]].to_numpy().tolist() == (daily / 24).to_numpy().tolist()

    def test_daily_to_hourly_intensive_repeats_each_day(self, kind):
        daily = as_kind(pd.Series([1.0, 2.0, 3.0], index=pd.date_range('2024-01-01', periods=3, freq='D')), kind)
        result = TimeSeriesGranularityConverter().convert(daily, '1h', QuantityTypeEnum.INTENSIVE)
        assert len(result) == 72
        assert result.iloc[[0, 30, 60]].to_numpy().tolist() == daily.to_numpy().tolist()


@pytest.mark.parametrize('target', ['D', '1D'])
def test_hourly_to_days_across_dst_gives_calendar_days(target):
    hourly = pd.Series(1.0, index=pd.date_range('2025-03-29', '2025-03-31 23:00', freq='h', tz='Europe/Berlin'))
    result = TimeSeriesGranularityConverter().convert(hourly, target, QuantityTypeEnum.EXTENSIVE)
    assert result.tolist() == [24.0, 23.0, 24.0]
    assert [timestamp.day for timestamp in result.index] == [29, 30, 31]


def test_time_level_converts_per_combination_of_the_other_levels():
    index = pd.date_range('2024-01-01', periods=4, freq='15min', name='datetime')
    data = pd.concat(
        {'a': pd.Series([1.0, 2.0, 3.0, 4.0], index=index), 'b': pd.Series(2.0, index=index)}, names=['dataset']
    )
    result = TimeSeriesGranularityConverter().convert(data, '1h', QuantityTypeEnum.INTENSIVE, time_level='datetime')
    assert result.index.names == ['dataset', 'datetime']
    assert result.tolist() == [2.5, 2.0]


@pytest.mark.parametrize(
    'value, expected',
    [
        ('h', pd.Timedelta('1h')),
        ('D', pd.Timedelta('1D')),
        ('15min', pd.Timedelta('15min')),
        ('15 minutes', pd.Timedelta('15min')),
        (pd.Timedelta('2h'), pd.Timedelta('2h')),
    ],
)
def test_parse_frequency_accepts_fixed_frequencies(value, expected):
    assert pd.Timedelta(TimeSeriesGranularityConverter.parse_frequency(value)) == expected


@pytest.mark.parametrize('value', ['hourly', 'W', 'ME', '0min', '-1h', 3600, 1.5, None])
def test_parse_frequency_rejects_other_values(value):
    with pytest.raises(ValueError, match='is not a positive fixed frequency'):
        TimeSeriesGranularityConverter.parse_frequency(value)


class TestWholeDaysAreCalendarDays:
    BERLIN = 'Europe/Berlin'

    @pytest.mark.parametrize('value', ['24h', '1 day', pd.Timedelta('1D')])
    def test_parse_frequency_maps_whole_days_to_calendar_days(self, value):
        assert TimeSeriesGranularityConverter.parse_frequency(value) == pd.offsets.Day(1)

    def test_a_timedelta_of_one_day_gives_the_same_result_as_d(self):
        hourly = pd.Series(1.0, index=pd.date_range('2024-03-30', periods=71, freq='h', tz=self.BERLIN))
        by_timedelta = TimeSeriesGranularityConverter().convert(hourly, pd.Timedelta('1D'), QuantityTypeEnum.EXTENSIVE)
        by_alias = TimeSeriesGranularityConverter().convert(hourly, 'D', QuantityTypeEnum.EXTENSIVE)
        pd.testing.assert_series_equal(by_timedelta, by_alias)
        assert by_alias.tolist() == [24.0, 23.0, 24.0]

    def test_convert_to_target_index_across_dst_gives_calendar_days(self):
        hourly = pd.Series(1.0, index=pd.date_range('2024-03-30', periods=71, freq='h', tz=self.BERLIN))
        target = pd.date_range('2024-03-30', periods=3, freq='D', tz=self.BERLIN)
        result = TimeSeriesGranularityConverter().convert_to_target_index(hourly, target, QuantityTypeEnum.EXTENSIVE)
        assert result.tolist() == [24.0, 23.0, 24.0]


class TestSubDailyStepsAreAbsolute:
    """Across a DST change, hourly data has a step of one hour, not of two or zero wall-clock hours."""

    def test_spring_forward_from_01_00_to_03_00(self):
        index = pd.DatetimeIndex(['2024-03-31 01:00', '2024-03-31 03:00']).tz_localize('Europe/Berlin')
        result = TimeSeriesGranularityConverter().convert(
            pd.Series(4.0, index=index), '15min', QuantityTypeEnum.EXTENSIVE
        )
        assert len(result) == 8
        assert (result == 1.0).all()

    def test_fall_back_from_02_00_cest_to_02_00_cet(self):
        index = (
            pd.DatetimeIndex(['2024-10-27 00:00', '2024-10-27 01:00']).tz_localize('UTC').tz_convert('Europe/Berlin')
        )
        assert [stamp.hour for stamp in index] == [2, 2]
        result = TimeSeriesGranularityConverter().convert(
            pd.Series(4.0, index=index), '15min', QuantityTypeEnum.EXTENSIVE
        )
        assert len(result) == 8
        assert result.index.is_unique
        assert (result == 1.0).all()


def test_nat_in_the_index_raises():
    index = pd.DatetimeIndex(['2024-01-01 00:00', None, '2024-01-01 01:00'])
    with pytest.raises(ValueError, match='NaT'):
        TimeSeriesGranularityConverter().convert(pd.Series(1.0, index=index), '15min', QuantityTypeEnum.INTENSIVE)


def test_time_level_keeps_groups_with_a_missing_key():
    index = pd.date_range('2024-01-01', periods=2, freq='h', name='datetime')
    keys = pd.Index(['a', None], dtype=object).repeat(2)
    data = pd.Series(4.0, index=pd.MultiIndex.from_arrays([keys, index.append(index)], names=['dataset', 'datetime']))
    result = TimeSeriesGranularityConverter().convert(data, '15min', QuantityTypeEnum.EXTENSIVE, time_level='datetime')
    assert len(result) == 16
    assert result.index.get_level_values('dataset').isna().sum() == 8


class TestSpansOfTheWholeInput:
    """A repeated row never reaches past the next input row, also across a change of direction."""

    @staticmethod
    def check(data, target):
        result = TimeSeriesGranularityConverter().convert(data, target, QuantityTypeEnum.EXTENSIVE)
        assert result.index.is_unique
        assert result.sum() == pytest.approx(data.sum())
        return result

    def test_four_hourly_day_then_hourly_day_to_hourly(self):
        data = pd.concat(
            [
                pd.Series(4.0, index=pd.date_range('2024-01-01', periods=6, freq='4h')),
                pd.Series(1.0, index=pd.date_range('2024-01-02', periods=24, freq='h')),
            ]
        )
        result = self.check(data, '1h')
        assert len(result) == 48
        assert (result == 1.0).all()

    def test_hourly_at_half_past_then_quarter_hourly_day_to_quarter_hours(self):
        data = pd.concat(
            [
                pd.Series(4.0, index=pd.date_range('2024-01-01 00:30', periods=24, freq='h')),
                pd.Series(1.0, index=pd.date_range('2024-01-02 00:15', periods=95, freq='15min')),
            ]
        )
        result = self.check(data, '15min')
        assert result.index.max() == pd.Timestamp('2024-01-02 23:45')


class TestDaysWithOneRow:
    """A day with one row takes the step to the next input row, at most one day."""

    def test_hourly_then_daily_rows_to_quarter_hours(self):
        hourly = pd.Series(1.0, index=pd.date_range('2024-01-01', periods=48, freq='h'))
        daily = pd.Series(24.0, index=pd.date_range('2024-01-03', periods=3, freq='D'))
        data = pd.concat([hourly, daily])
        result = TimeSeriesGranularityConverter().convert(data, '15min', QuantityTypeEnum.EXTENSIVE)
        assert result.groupby(result.index.date).size().tolist() == [96] * 5
        assert result.sum() == pytest.approx(data.sum())
        assert result.index.is_unique

    def test_daily_then_hourly_rows_to_hourly(self):
        daily = pd.Series(24.0, index=pd.date_range('2024-01-01', periods=3, freq='D'))
        hourly = pd.Series(1.0, index=pd.date_range('2024-01-04', periods=24, freq='h'))
        data = pd.concat([daily, hourly])
        result = TimeSeriesGranularityConverter().convert(data, '1h', QuantityTypeEnum.EXTENSIVE)
        assert result.groupby(result.index.date).size().tolist() == [24] * 4
        assert (result == 1.0).all()

    def test_a_single_row_before_a_gap_of_days_spreads_up_to_the_next_row(self):
        data = pd.concat(
            [
                pd.Series(1.0, index=pd.date_range('2024-01-01', periods=24, freq='h')),
                pd.Series(96.0, index=pd.DatetimeIndex(['2024-01-02'])),
                pd.Series(1.0, index=pd.date_range('2024-01-06', periods=24, freq='h')),
            ]
        )
        result = TimeSeriesGranularityConverter().convert(data, '1h', QuantityTypeEnum.EXTENSIVE)
        spread = result.loc['2024-01-02':'2024-01-05']
        assert len(spread) == 96
        assert (spread == 1.0).all()
        assert result.sum() == pytest.approx(data.sum())

    @pytest.mark.parametrize('tz', [None, 'Europe/Berlin'])
    def test_weekly_data_to_days(self, tz):
        data = pd.Series([7.0, 14.0, 21.0], index=pd.date_range('2024-03-04', periods=3, freq='7D', tz=tz))
        result = TimeSeriesGranularityConverter().convert(data, 'D', QuantityTypeEnum.EXTENSIVE)
        assert len(result) == 21
        assert result.tolist() == [1.0] * 7 + [2.0] * 7 + [3.0] * 7
        assert result.sum() == pytest.approx(data.sum())

    @pytest.mark.parametrize('tz', [None, 'Europe/Berlin'])
    def test_monthly_data_to_days(self, tz):
        data = pd.Series(1.0, index=pd.date_range('2024-01-01', periods=12, freq='MS', tz=tz))
        result = TimeSeriesGranularityConverter().convert(data, 'D', QuantityTypeEnum.EXTENSIVE)
        months = result.index.to_period('M') if tz is None else result.index.tz_localize(None).to_period('M')
        rows_per_month = result.groupby(months).size()
        assert rows_per_month.iloc[:-1].tolist() == list(data.index[:-1].days_in_month)
        assert result.groupby(months).sum().iloc[:-1].tolist() == pytest.approx([1.0] * 11)
        assert result.index.is_unique


@pytest.mark.parametrize(
    'data, time_level, error',
    [
        (pd.Series(1.0, index=pd.date_range('2024-01-01', periods=2, freq='h')), 'datetime', TypeError),
        (
            pd.Series(
                1.0,
                index=pd.MultiIndex.from_product(
                    [['a'], pd.date_range('2024-01-01', periods=2, freq='h')], names=['dataset', 'datetime']
                ),
            ),
            'time',
            ValueError,
        ),
        (
            pd.Series(
                1.0,
                index=pd.MultiIndex.from_product(
                    [['a'], pd.date_range('2024-01-01', periods=2, freq='h')], names=['dataset', 'datetime']
                ),
            ),
            2,
            ValueError,
        ),
    ],
)
def test_a_bad_time_level_raises(data, time_level, error):
    with pytest.raises(error, match='time_level'):
        TimeSeriesGranularityConverter().convert(data, '15min', QuantityTypeEnum.INTENSIVE, time_level=time_level)
