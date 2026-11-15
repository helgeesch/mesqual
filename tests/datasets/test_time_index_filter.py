import datetime

import pandas as pd
import pytest

from mesqual.datasets import DatasetConfig, DatasetLinkCollection, TimeIndexFilter, Range
from mesqual.cache import PickleCache
from tests.datasets.fixtures import Interpreter, Platform, StaticDataset, hourly_frame


def dates_of(data) -> list[datetime.date]:
    return sorted(set(data.index.get_level_values('datetime').date))


def fetch_filtered(data, *filters, flag='flag'):
    ds = StaticDataset({flag: data})
    return ds.fetch(flag, config={'time_index_filter': filters if len(filters) > 1 else filters[0]})


class TestDates:
    def test_include_only(self):
        result = fetch_filtered(hourly_frame(), TimeIndexFilter.dates(include=['2024-01-02']))
        assert dates_of(result) == [datetime.date(2024, 1, 2)]
        assert len(result) == 24

    def test_exclude_only(self):
        result = fetch_filtered(hourly_frame(), TimeIndexFilter.dates(exclude=[datetime.date(2024, 1, 2)]))
        assert dates_of(result) == [datetime.date(2024, 1, 1), datetime.date(2024, 1, 3)]

    def test_include_and_exclude(self):
        time_index_filter = TimeIndexFilter.dates(include=[Range('2024-01-01', '2024-01-03')], exclude=['2024-01-02'])
        result = fetch_filtered(hourly_frame(), time_index_filter)
        assert dates_of(result) == [datetime.date(2024, 1, 1), datetime.date(2024, 1, 3)]

    def test_range_of_dates(self):
        result = fetch_filtered(
            hourly_frame(periods=24 * 5), TimeIndexFilter.dates(include=[Range('2024-01-02', '2024-01-03')])
        )
        assert dates_of(result) == [datetime.date(2024, 1, 2), datetime.date(2024, 1, 3)]
        assert len(result) == 48

    def test_exact_timestamps(self):
        stamps = [pd.Timestamp('2024-01-01 05:00'), pd.Timestamp('2024-01-02 07:00')]
        result = fetch_filtered(hourly_frame(), TimeIndexFilter.dates(include=stamps))
        assert list(result.index) == stamps

    def test_range_of_timestamps_across_day_boundary(self):
        delivery_day = Range(pd.Timestamp('2024-01-01 23:00'), pd.Timestamp('2024-01-02 22:00'))
        result = fetch_filtered(hourly_frame(), TimeIndexFilter.dates(include=[delivery_day]))
        assert len(result) == 24
        assert result.index[0] == pd.Timestamp('2024-01-01 23:00')
        assert result.index[-1] == pd.Timestamp('2024-01-02 22:00')

    def test_time_zone_matches_local_calendar_date(self):
        data = hourly_frame(start='2024-01-01 00:00', periods=48, tz='Europe/Berlin')
        result = fetch_filtered(data, TimeIndexFilter.dates(include=['2024-01-02']))
        assert len(result) == 24
        assert all(t.date() == datetime.date(2024, 1, 2) for t in result.index)

    def test_dst_day_has_23_hours(self):
        data = hourly_frame(start='2024-03-30 00:00', periods=72, tz='Europe/Berlin')
        result = fetch_filtered(data, TimeIndexFilter.dates(include=['2024-03-31']))
        assert len(result) == 23

    def test_dst_day_without_midnight(self):
        data = hourly_frame(start='2024-09-07 00:00', periods=72, tz='America/Santiago')
        result = fetch_filtered(data, TimeIndexFilter.dates(include=['2024-09-08']))
        assert len(result) == 23
        assert all(t.date() == datetime.date(2024, 9, 8) for t in result.index)
        result = fetch_filtered(data, TimeIndexFilter.dates(include=[Range('2024-09-08', '2024-09-09')]))
        assert len(result) == 47

    def test_datetime_level_of_row_multiindex(self):
        base = hourly_frame()
        data = pd.concat({'run1': base, 'run2': base}, names=['run'])
        result = fetch_filtered(data, TimeIndexFilter.dates(include=['2024-01-03']))
        assert len(result) == 48
        assert result.index.names == ['run', 'datetime']
        assert dates_of(result) == [datetime.date(2024, 1, 3)]

    def test_series(self):
        result = fetch_filtered(hourly_frame()['a'], TimeIndexFilter.dates(include=['2024-01-01']))
        assert isinstance(result, pd.Series)
        assert len(result) == 24


class TestOtherLevels:
    @staticmethod
    def day_hour_frame() -> pd.DataFrame:
        index = pd.MultiIndex.from_product([['2024-01-01', '2024-01-02'], range(1, 25)], names=['day', 'hour'])
        return pd.DataFrame({'a': range(len(index))}, index=index)

    def test_one_level_of_multiindex_keeps_other_levels(self):
        result = fetch_filtered(self.day_hour_frame(), TimeIndexFilter(level='hour', include=[Range(1, 4)]))
        assert result.index.names == ['day', 'hour']
        assert len(result) == 8
        assert set(result.index.get_level_values('hour')) == {1, 2, 3, 4}

    def test_two_filters_on_two_levels_combine_with_and(self):
        result = fetch_filtered(
            self.day_hour_frame(),
            TimeIndexFilter(level='hour', include=[Range(1, 4)]),
            TimeIndexFilter(level='day', include=['2024-01-02']),
        )
        assert list(result.index) == [('2024-01-02', h) for h in range(1, 5)]

    def test_years_with_values_and_range(self):
        data = pd.DataFrame({'a': range(6)}, index=pd.Index(range(2025, 2031), name='year'))
        result = fetch_filtered(data, TimeIndexFilter(level='year', include=[2025, Range(2028, 2029)]))
        assert list(result.index) == [2025, 2028, 2029]

    def test_tuple_periods_with_values_and_range(self):
        periods = pd.Index([(1, 1), (1, 2), (2, 1), (2, 2)], tupleize_cols=False, name='period')
        data = pd.DataFrame({'a': range(4)}, index=periods)
        result = fetch_filtered(data, TimeIndexFilter(level='period', include=[Range((1, 2), (2, 1))]))
        assert list(result.index) == [(1, 2), (2, 1)]
        result = fetch_filtered(data, TimeIndexFilter(level='period', exclude=[(1, 1)]))
        assert list(result.index) == [(1, 2), (2, 1), (2, 2)]

    def test_result_without_level_is_unchanged(self):
        data = hourly_frame()
        result = fetch_filtered(data, TimeIndexFilter(level='period', include=[1]))
        pd.testing.assert_frame_equal(result, data)

    def test_time_level_filter_on_model_frame_does_nothing(self):
        model = pd.DataFrame({'zone': ['A', 'B']}, index=pd.Index(['n1', 'n2'], name='node'))
        result = fetch_filtered(model, TimeIndexFilter.dates(include=['2024-01-01']))
        pd.testing.assert_frame_equal(result, model)

    def test_several_filters_apply_in_order(self):
        result = fetch_filtered(
            hourly_frame(),
            TimeIndexFilter.dates(include=[Range('2024-01-01', '2024-01-02')]),
            TimeIndexFilter.dates(exclude=['2024-01-01']),
        )
        assert dates_of(result) == [datetime.date(2024, 1, 2)]


class TestOutermostFetchOnly:
    def test_child_filter_has_no_effect_through_collection(self):
        child = StaticDataset(
            {'flag': hourly_frame()},
            name='child',
            config=DatasetConfig(time_index_filter=TimeIndexFilter.dates(include=['2024-01-01'])),
        )
        collection = DatasetLinkCollection([child], name='collection')
        assert len(collection.fetch('flag')) == 72
        assert len(child.fetch('flag')) == 24

    def test_collection_filter_applies_once_at_the_top(self):
        child = StaticDataset({'flag': hourly_frame()}, name='child')
        collection = DatasetLinkCollection([child], name='collection')
        result = collection.fetch('flag', config={'time_index_filter': TimeIndexFilter.dates(include=['2024-01-02'])})
        assert len(result) == 24

    def test_cached_platform_dataset_holds_one_unfiltered_entry(self, tmp_path):
        class CachedPlatform(Platform):
            pass

        @CachedPlatform.register_interpreter
        class PriceInterpreter(Interpreter):
            FLAGS = frozenset({'price', 'price_mean'})

            def _fetch(self, flag, effective_config, **kwargs):
                if flag == 'price':
                    return self.prices
                return self.parent_dataset.fetch('price', effective_config).mean(axis=1).to_frame('mean')

        cache = PickleCache(str(tmp_path))
        ds = CachedPlatform(name='scenario', cache=cache, prices=hourly_frame())
        first = ds.fetch('price', config={'time_index_filter': TimeIndexFilter.dates(include=['2024-01-01'])})
        second = ds.fetch(
            'price', config={'time_index_filter': TimeIndexFilter.dates(include=['2024-01-02', '2024-01-03'])}
        )
        assert dates_of(first) == [datetime.date(2024, 1, 1)]
        assert dates_of(second) == [datetime.date(2024, 1, 2), datetime.date(2024, 1, 3)]
        (key,) = cache.list_keys(ds, 'price')
        _, value = cache.entry(key)
        assert len(value) == 72

    def test_filtered_and_unfiltered_fetch_share_one_cache_entry(self, tmp_path):
        ds = StaticDataset({'flag': hourly_frame()}, name='scenario', cache=PickleCache(str(tmp_path)))
        filtered = ds.fetch('flag', config={'time_index_filter': TimeIndexFilter.dates(include=['2024-01-01'])})
        unfiltered = ds.fetch('flag')
        assert len(filtered) == 24
        assert len(unfiltered) == 72
        assert len(ds.seen_configs) == 1
        assert len(list(tmp_path.iterdir())) == 1

    def test_derived_flag_with_instance_filter_caches_unfiltered_data(self, tmp_path):
        class CachedPlatform(Platform):
            pass

        @CachedPlatform.register_interpreter
        class PriceInterpreter(Interpreter):
            FLAGS = frozenset({'price', 'price_mean'})

            def _fetch(self, flag, effective_config, **kwargs):
                if flag == 'price':
                    return self.prices
                return self.parent_dataset.fetch('price', effective_config).mean(axis=1).to_frame('mean')

        config = DatasetConfig(time_index_filter=TimeIndexFilter.dates(include=['2024-01-01']))
        cache = PickleCache(str(tmp_path))
        ds = CachedPlatform(name='scenario', cache=cache, config=config, prices=hourly_frame())
        assert len(ds.fetch('price_mean')) == 24
        for key in cache.list_keys():
            _, value = cache.entry(key)
            assert len(value) == 72


def test_invalid_date_value_raises():
    with pytest.raises(TypeError):
        fetch_filtered(hourly_frame(), TimeIndexFilter.dates(include=[3.5]))


class TestInvalidDatetimeValues:
    def test_range_mixing_timestamp_and_date_raises(self):
        with pytest.raises(TypeError, match='two dates or two Timestamps'):
            fetch_filtered(
                hourly_frame(), TimeIndexFilter.dates(include=[Range(pd.Timestamp('2024-01-01 05:00'), '2024-01-02')])
            )

    def test_tz_aware_timestamp_on_naive_index_raises(self):
        with pytest.raises(TypeError, match='tz-aware'):
            fetch_filtered(hourly_frame(), TimeIndexFilter.dates(include=[pd.Timestamp('2024-01-01 05:00', tz='UTC')]))

    def test_naive_timestamp_on_tz_aware_index_raises(self):
        with pytest.raises(TypeError, match='tz-aware'):
            fetch_filtered(
                hourly_frame(tz='Europe/Berlin'), TimeIndexFilter.dates(include=[pd.Timestamp('2024-01-01 05:00')])
            )
