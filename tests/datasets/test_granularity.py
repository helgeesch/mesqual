"""Generic granularity conversion on the outermost fetch."""

import re

import pandas as pd
import pytest

from mesqual.cache import PickleCache
from mesqual.datasets import (
    INHERIT,
    DatasetComparison,
    DatasetConcatCollection,
    DatasetConcatCollectionOfComparisons,
    DatasetConfig,
    DatasetConfigManager,
)
from mesqual.energy_data_handling import TimeSeriesGranularityConverter
from mesqual.enums import QuantityTypeEnum
from mesqual.flag import Dimension
from mesqual.testing import DummyDataset, DummyScenario
from mesqual.units import Units
from tests.datasets.fixtures import PlatformStaticDataset, StaticDataset, series_spec


def quarter_hourly() -> pd.DataFrame:
    index = pd.date_range('2024-01-01', periods=8, freq='15min', name='datetime')
    return pd.DataFrame({'a': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]}, index=index)


def dataset(unit=Units.MW, **kwargs) -> StaticDataset:
    return StaticDataset({'x': quarter_hourly()}, specs=(series_spec('x', unit=unit),), **kwargs)


class TestFetch:
    def test_intensive_flag_is_averaged(self):
        assert dataset().fetch('x', granularity='1h')['a'].tolist() == [2.5, 6.5]

    def test_extensive_flag_is_summed(self):
        assert dataset(Units.MWh).fetch('x', granularity='1h')['a'].tolist() == [10.0, 26.0]

    def test_class_config(self):
        DatasetConfigManager.set_class_config(StaticDataset, DatasetConfig(granularity='30min'))
        assert dataset().fetch('x')['a'].tolist() == [1.5, 3.5, 5.5, 7.5]

    def test_instance_config(self):
        assert dataset(config=DatasetConfig(granularity='30min')).fetch('x')['a'].tolist() == [1.5, 3.5, 5.5, 7.5]

    def test_fetch_time_config(self):
        assert dataset().fetch('x', config={'granularity': '30min'})['a'].tolist() == [1.5, 3.5, 5.5, 7.5]

    def test_fetch_argument_wins_over_the_config_layers(self):
        ds = dataset(config=DatasetConfig(granularity='30min'))
        assert ds.fetch('x', granularity='1h')['a'].tolist() == [2.5, 6.5]
        assert ds.fetch('x', config={'granularity': '30min'}, granularity='1h')['a'].tolist() == [2.5, 6.5]

    def test_explicit_none_gives_the_native_resolution(self):
        ds = dataset(config=DatasetConfig(granularity='1h'))
        pd.testing.assert_frame_equal(ds.fetch('x', granularity=None), quarter_hourly())
        pd.testing.assert_frame_equal(ds.fetch('x', config={'granularity': None}), quarter_hourly())

    def test_param_defaults_entry_raises(self):
        with pytest.raises(ValueError, match=r"does not declare: \['granularity'\]"):
            dataset().fetch('x', config={'param_defaults': {'granularity': '1h'}})

    def test_granularity_is_no_dataset_parameter(self):
        assert 'granularity' not in {d.name for d in StaticDataset.fetch_params}
        assert 'granularity' not in {p.name for p in dataset().flag_info('x').params}

    def test_hourly_to_quarter_hours_on_a_dst_day(self):
        index = pd.date_range('2025-03-30', periods=23, freq='h', tz='Europe/Berlin', name='datetime')
        hourly = pd.DataFrame({'a': range(1, 24)}, index=index, dtype=float)
        ds = StaticDataset({'x': hourly}, specs=(series_spec('x', unit=Units.MWh),))
        result = ds.fetch('x', granularity='15min')
        assert len(result) == 92
        assert result['a'].sum() == pytest.approx(hourly['a'].sum())

    def test_without_granularity_the_native_resolution(self):
        pd.testing.assert_frame_equal(dataset().fetch('x'), quarter_hourly())

    @pytest.mark.parametrize('value', ['hourly', 3600, 1.5, '0min', '-1h', 'W'])
    def test_invalid_value_raises(self, value):
        with pytest.raises(
            ValueError, match=f'granularity: {re.escape(repr(value))} is not a positive fixed frequency'
        ):
            dataset().fetch('x', granularity=value)

    def test_invalid_value_in_the_instance_config_raises_at_fetch(self):
        with pytest.raises(ValueError, match='granularity'):
            dataset(config=DatasetConfig(granularity='hourly')).fetch('x')

    def test_invalid_value_of_a_child_has_no_effect_through_a_collection(self):
        child = dataset(name='bad', config=DatasetConfig(granularity='hourly'))
        result = DatasetConcatCollection([child, dataset(name='ok')]).fetch('x')
        assert len(result) == 8

    def test_value_of_a_child_has_no_effect_through_a_collection(self):
        child = dataset(name='a', config=DatasetConfig(granularity='1h'))
        assert len(DatasetConcatCollection([child]).fetch('x')) == 8

    def test_argument_on_a_nested_fetch_keeps_the_inherited_config(self, tmp_path):
        class Nested(PlatformStaticDataset):
            def _fetch(self, flag, effective_config, **kwargs):
                if flag == 'top':
                    return self.fetch('x', effective_config, granularity=None)
                return quarter_hourly() * (1.0 if effective_config.convert_periods else 2.0)

        ds = Nested({'x': None, 'top': None}, name='n', cache=PickleCache(str(tmp_path)))
        ds.fetch('top', config={'convert_periods': True})
        result = ds.fetch('top', config={'convert_periods': False})
        pd.testing.assert_frame_equal(result, quarter_hourly() * 2.0)
        (dependency,) = ds.get_fetch_record('top', config={'convert_periods': False}).dependencies
        assert dependency.call_config is INHERIT

    def test_time_index_filter_and_granularity_in_one_config_filter_first(self):
        from mesqual.datasets import Range, TimeIndexFilter

        window = TimeIndexFilter(include=[Range(pd.Timestamp('2024-01-01 00:30'), pd.Timestamp('2024-01-01 01:15'))])
        ds = dataset(config=DatasetConfig(time_index_filter=window, granularity='1h'))
        assert ds.fetch('x')['a'].tolist() == [3.5, 5.5]

    def test_fetch_merged_takes_the_argument(self):
        collection = DatasetConcatCollection([dataset(name='a'), dataset(name='b')])
        assert collection.fetch_merged('x', granularity='1h')['a'].tolist() == [2.5, 6.5]

    def test_get_fetch_record_with_config_and_argument_finds_the_record_of_one_config(self):
        ds = dataset()
        ds.fetch('x', config=DatasetConfig(use_cache=False, granularity='1h'))
        record = ds.get_fetch_record('x', config={'use_cache': False}, granularity='1h')
        assert record is not None and record.call_config == {'use_cache': False, 'granularity': '1h'}

    def test_hourly_to_days_across_dst_on_a_dataset(self):
        index = pd.date_range('2025-03-29', '2025-03-31 23:00', freq='h', tz='Europe/Berlin', name='datetime')
        ds = StaticDataset({'x': pd.DataFrame({'a': 1.0}, index=index)}, specs=(series_spec('x', unit=Units.MWh),))
        assert ds.fetch('x', granularity='D')['a'].tolist() == [24.0, 23.0, 24.0]

    @pytest.mark.parametrize('value, rows', [('h', 2), ('D', 1), (pd.Timedelta('30min'), 4)])
    def test_frequency_strings_and_timedeltas(self, value, rows):
        assert len(dataset().fetch('x', granularity=value)) == rows

    def test_non_datetime_index_raises(self):
        frame = quarter_hourly().reset_index(drop=True)
        with pytest.raises(TypeError, match='DatetimeIndex'):
            StaticDataset({'x': frame}).fetch('x', granularity='1h')

    def test_model_flag_is_unchanged(self):
        model = DummyDataset(DummyScenario(name='base', days=1))
        pd.testing.assert_frame_equal(model.fetch('Node.Model', granularity='1h'), model.fetch('Node.Model'))

    def test_applies_after_the_time_index_filter(self):
        from mesqual.datasets import Range, TimeIndexFilter

        window = TimeIndexFilter(include=[Range(pd.Timestamp('2024-01-01 00:30'), pd.Timestamp('2024-01-01 01:15'))])
        result = dataset().fetch('x', config={'time_index_filter': window}, granularity='1h')
        assert result['a'].tolist() == [3.5, 5.5]


class TestOutermostOnly:
    @pytest.mark.parametrize(
        'route',
        [
            lambda value: {'granularity': value},
            lambda value: {'config': {'granularity': value}},
            None,
        ],
        ids=['argument', 'config', 'instance_config'],
    )
    def test_cache_holds_one_native_entry(self, tmp_path, route):
        cache = PickleCache(str(tmp_path))
        dataset(name='a', cache=cache).fetch('x')
        for value, rows in (('1h', 2), ('30min', 4), ('2h', 1)):
            if route is None:
                result = dataset(name='a', cache=cache, config=DatasetConfig(granularity=value)).fetch('x')
            else:
                result = dataset(name='a', cache=cache).fetch('x', **route(value))
            assert len(result) == rows
        (key,) = cache.list_keys()
        _, value = cache.entry(key)
        pd.testing.assert_frame_equal(value, quarter_hourly())

    def test_child_keeps_its_own_param_defaults_with_both_granularity_routes(self):
        solution = Dimension('solution', 'Solutions.', {'final': 'final', 'first': 'first'}, default='final')

        class Solutions(StaticDataset):
            def _fetch(self, flag, effective_config, **params):
                return super()._fetch(flag, effective_config) * (10 if params['solution'] == 'first' else 1)

        def collection():
            child = Solutions(
                {'x': quarter_hourly()},
                name='a',
                specs=(series_spec('x', params=(solution,)),),
                config=DatasetConfig(param_defaults={'solution': 'first'}),
            )
            return DatasetConcatCollection([child])

        by_argument = collection().fetch('x', granularity='1h')
        by_config = collection().fetch('x', config={'granularity': '1h'})
        pd.testing.assert_frame_equal(by_config, by_argument)
        assert by_argument[('a', 'a')].tolist() == [25.0, 65.0]

    def test_granularity_is_not_a_parameter_of_fetch_or_of_the_record(self):
        ds = dataset()
        ds.fetch('x', granularity='1h')
        record = ds.get_fetch_record('x', granularity='1h')
        assert record.call_kwargs == {}
        assert record.call_config == {'granularity': '1h'}
        assert 'granularity' not in record.params
        assert ds.seen_configs[-1].granularity is None

    def test_collection_converts_once_and_children_get_native_data(self):
        children = [dataset(name='a'), dataset(name='b')]
        collection = DatasetConcatCollection(children)
        result = collection.fetch('x', granularity='1h')
        assert result[('a', 'a')].tolist() == [2.5, 6.5]
        record = collection.get_fetch_record('x', granularity='1h')
        assert all(dependency.call_kwargs == {} for dependency in record.dependencies)
        assert all(child.seen_configs[-1].granularity is None for child in children)

    def test_derived_flag_aggregates_native_values_and_converts_once(self):
        scenario = DummyScenario(name='base', days=1, resolution_change=('2024-01-01', '15min'))
        ds = DummyDataset(scenario)
        native = ds.fetch('Node.Results.demand_total')
        expected = TimeSeriesGranularityConverter().convert(native, '1h', QuantityTypeEnum.INTENSIVE)
        pd.testing.assert_frame_equal(ds.fetch('Node.Results.demand_total', granularity='1h'), expected)
        record = ds.get_fetch_record('Node.Results.demand_total', granularity='1h')
        assert all('granularity' not in d.call_kwargs for d in record.iter_dependencies())
        assert all(d.config_reads.get('granularity') is None for d in record.iter_dependencies())

    @pytest.mark.parametrize('route', [{'granularity': '1h'}, {'config': {'granularity': '1h'}}])
    def test_concat_on_rows_converts_the_datetime_level(self, route):
        collection = DatasetConcatCollection([dataset(name='a'), dataset(name='b')], concat_axis=0)
        result = collection.fetch('x', **route)
        assert result.index.names == ['dataset', 'datetime']
        assert result.loc['b', 'a'].tolist() == [2.5, 6.5]
        assert len(result) == 4

    def test_empty_concat_on_rows_is_returned_unchanged(self):
        empty = quarter_hourly().iloc[:0]
        children = [StaticDataset({'x': empty}, name=name, specs=(series_spec('x'),)) for name in ('a', 'b')]
        result = DatasetConcatCollection(children, concat_axis=0).fetch('x', granularity='1h')
        assert result.empty
        assert result.index.names == ['dataset', 'datetime']

    def test_nested_concat_on_rows_converts_the_datetime_level(self):
        base, high = (
            dataset(name='base'),
            StaticDataset({'x': quarter_hourly() * 2}, name='high', specs=(series_spec('x'),)),
        )
        scen = DatasetConcatCollection([base, high], name='scenario', concat_axis=0)
        comp = DatasetConcatCollectionOfComparisons([DatasetComparison(high, base)], name='comparison', concat_axis=0)
        both = DatasetConcatCollection([scen, comp], concat_level_name='type', concat_axis=0)
        result = both.fetch('x', granularity='1h')
        assert result.index.names == ['type', 'dataset', 'datetime']
        assert result.xs(('comparison', 'high vs base'), level=['type', 'dataset'])['a'].tolist() == [2.5, 6.5]
        assert result.xs(('scenario', 'high'), level=['type', 'dataset'])['a'].tolist() == [5.0, 13.0]

    def test_fetch_multiple_flags_converts_each_flag(self):
        ds = StaticDataset({'x': quarter_hourly(), 'y': quarter_hourly() * 2})
        result = ds.fetch_multiple_flags_and_concat(['x', 'y'], granularity='1h')
        assert result[('y', 'a')].tolist() == [5.0, 13.0]
