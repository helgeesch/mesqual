import copy
import dataclasses
import logging
import pickle

import pandas as pd
import pytest

from dataclasses import dataclass

from mesqual.datasets import DatasetConfig, DatasetConfigManager, DatasetLinkCollection, TimeIndexFilter, config_field
from mesqual.enums import TopologyTypeEnum
from mesqual.flag import ModelFlagSpec
from tests.datasets.fixtures import (
    Interpreter,
    Platform,
    PlatformConfig,
    PlatformStaticDataset,
    StaticDataset,
    StudyConfig,
    StudyStaticDataset,
    hourly_frame,
)


FIRST_DAY = TimeIndexFilter.dates(include=['2024-01-01'])


@dataclass
class NoCacheConfig(DatasetConfig):
    use_cache: bool = config_field(False, 'Read from and write to the cache of the dataset.')


@dataclass
class KeywordOnlyConfig(DatasetConfig):
    tag: str = config_field('', 'Keyword-only tag.', kw_only=True)
    level: int = config_field(0, 'Positional level.')


class NoCacheDataset(StaticDataset):
    @classmethod
    def get_config_type(cls):
        return NoCacheConfig


def make_dataset(cls=StaticDataset, **kwargs):
    return cls({'flag': hourly_frame()}, **kwargs)


class TestExplicitFields:
    def test_constructor_keywords_are_explicit(self):
        assert DatasetConfig(use_cache=False).explicit_fields == {'use_cache'}

    def test_positional_arguments_map_to_field_names(self):
        assert DatasetConfig(False, False).explicit_fields == {'use_cache', 'auto_sort_index'}

    def test_subclass_fields_keep_their_positions(self):
        config = PlatformConfig(True, True, 'last', None, None, False)
        assert config.convert_periods is False and config.granularity is None
        assert 'granularity' not in config.explicit_fields

    def test_positional_arguments_skip_keyword_only_fields(self):
        config = KeywordOnlyConfig(True, True, 'last', None, None, 5)
        assert config.level == 5
        assert config.explicit_fields == {
            'use_cache',
            'auto_sort_index',
            'keep_duplicate_indices',
            'time_index_filter',
            'param_defaults',
            'level',
        }

    def test_default_construction_has_no_explicit_fields(self):
        assert DatasetConfig().explicit_fields == frozenset()

    def test_assignment_after_construction_is_explicit(self):
        config = DatasetConfig()
        config.keep_duplicate_indices = 'all'
        assert config.explicit_fields == {'keep_duplicate_indices'}

    def test_merge_gives_union_of_explicit_fields(self):
        merged = DatasetConfig(use_cache=False).merge(DatasetConfig(auto_sort_index=False))
        assert merged.explicit_fields == {'use_cache', 'auto_sort_index'}

    def test_merge_with_dict_marks_its_keys(self):
        merged = DatasetConfig().merge({'use_cache': False})
        assert merged.explicit_fields == {'use_cache'}

    def test_copy_keeps_explicit_fields(self):
        config = DatasetConfig(use_cache=False)
        assert copy.copy(config).explicit_fields == {'use_cache'}
        assert copy.deepcopy(config).explicit_fields == {'use_cache'}

    def test_pickle_keeps_explicit_fields(self):
        config = StudyConfig(apply_correction=True)
        restored = pickle.loads(pickle.dumps(config))
        assert restored == config
        assert restored.explicit_fields == {'apply_correction'}

    def test_dataclasses_replace_marks_every_field(self):
        replaced = dataclasses.replace(DatasetConfig(), use_cache=False)
        assert replaced.explicit_fields == {f.name for f in dataclasses.fields(DatasetConfig)}

    def test_explicit_fields_is_read_only(self):
        with pytest.raises(AttributeError):
            DatasetConfig().explicit_fields = {'use_cache'}


class TestMerge:
    def test_partial_override_keeps_lower_values(self):
        lower = DatasetConfig(use_cache=False, auto_sort_index=False)
        merged = lower.merge(DatasetConfig(keep_duplicate_indices='all'))
        assert merged.use_cache is False
        assert merged.auto_sort_index is False
        assert merged.keep_duplicate_indices == 'all'

    def test_explicit_none_overrides(self):
        merged = DatasetConfig(time_index_filter=FIRST_DAY).merge(DatasetConfig(time_index_filter=None))
        assert merged.time_index_filter is None
        assert DatasetConfig(time_index_filter=FIRST_DAY).merge({'time_index_filter': None}).time_index_filter is None

    def test_unset_field_does_not_override(self):
        merged = DatasetConfig(time_index_filter=FIRST_DAY).merge(DatasetConfig(use_cache=False))
        assert merged.time_index_filter == FIRST_DAY

    def test_clear_resets_to_class_default(self):
        merged = DatasetConfig(use_cache=False).merge(DatasetConfig(use_cache=DatasetConfig.CLEAR))
        assert merged.use_cache is True
        assert 'use_cache' in merged.explicit_fields

    def test_clear_survives_pickle_as_the_same_marker(self):
        assert pickle.loads(pickle.dumps(DatasetConfig.CLEAR)) is DatasetConfig.CLEAR

    def test_result_has_type_of_lower_layer(self):
        merged = StudyConfig().merge(DatasetConfig(use_cache=False))
        assert type(merged) is StudyConfig
        assert DatasetConfig().merge(StudyConfig(apply_correction=True)).__class__ is DatasetConfig

    def test_subclass_fields_survive_every_merge(self):
        study = StudyConfig(apply_correction=True)
        through_base = DatasetConfig().merge(study)
        back = StudyConfig().merge(through_base)
        assert back.apply_correction is True
        assert type(back) is StudyConfig

    def test_merge_does_not_change_inputs(self):
        lower = DatasetConfig(use_cache=False)
        lower.merge({'auto_sort_index': False})
        assert lower.auto_sort_index is True
        assert lower.explicit_fields == {'use_cache'}


class TestLayers:
    def test_layer_order(self):
        DatasetConfigManager.set_class_config(PlatformStaticDataset, PlatformConfig(convert_periods=False))
        DatasetConfigManager.set_class_config(StudyStaticDataset, StudyConfig(apply_correction=True))
        ds = make_dataset(StudyStaticDataset, config=StudyConfig(use_cache=False))
        ds.fetch('flag', config={'keep_duplicate_indices': 'all'})
        effective = ds.seen_configs[-1]
        assert type(effective) is StudyConfig
        assert effective.convert_periods is False
        assert effective.apply_correction is True
        assert effective.use_cache is False
        assert effective.keep_duplicate_indices == 'all'
        assert effective.auto_sort_index is True

    def test_subclass_class_config_wins_over_parent_class_config(self):
        DatasetConfigManager.set_class_config(PlatformStaticDataset, PlatformConfig(convert_periods=False))
        DatasetConfigManager.set_class_config(StudyStaticDataset, StudyConfig(convert_periods=True))
        assert make_dataset(StudyStaticDataset).instance_config.convert_periods is True

    def test_partial_instance_config_keeps_class_config(self):
        DatasetConfigManager.set_class_config(StaticDataset, DatasetConfig(auto_sort_index=False))
        ds = make_dataset(config=DatasetConfig(use_cache=False))
        assert ds.instance_config.auto_sort_index is False
        assert ds.instance_config.use_cache is False

    def test_fetch_time_dict_overrides_only_its_keys(self):
        DatasetConfigManager.set_class_config(StaticDataset, DatasetConfig(auto_sort_index=False))
        ds = make_dataset()
        ds.fetch('flag', config={'use_cache': False})
        assert ds.seen_configs[-1].auto_sort_index is False
        assert ds.seen_configs[-1].use_cache is False

    def test_fetch_time_config_object_overrides_only_its_explicit_fields(self):
        DatasetConfigManager.set_class_config(StaticDataset, DatasetConfig(auto_sort_index=False))
        ds = make_dataset()
        ds.fetch('flag', config=DatasetConfig(use_cache=False))
        assert ds.seen_configs[-1].auto_sort_index is False

    def test_fetch_time_none_removes_instance_filter(self):
        ds = make_dataset(config=DatasetConfig(time_index_filter=FIRST_DAY))
        assert len(ds.fetch('flag')) == 24
        assert len(ds.fetch('flag', config={'time_index_filter': None})) == 72
        assert len(ds.fetch('flag', config=DatasetConfig(time_index_filter=None))) == 72

    def test_fetch_time_config_without_filter_keeps_instance_filter(self):
        ds = make_dataset(config=DatasetConfig(time_index_filter=FIRST_DAY))
        assert len(ds.fetch('flag', config={'use_cache': False})) == 24

    def test_clear_at_fetch_time_resets_class_config(self):
        DatasetConfigManager.set_class_config(StaticDataset, DatasetConfig(auto_sort_index=False))
        ds = make_dataset()
        ds.fetch('flag', config={'auto_sort_index': DatasetConfig.CLEAR})
        assert ds.seen_configs[-1].auto_sort_index is True


class TestDuplicateIndices:
    @staticmethod
    def duplicated_dataset(name: str = None) -> StaticDataset:
        index = pd.DatetimeIndex(['2024-01-01 00:00', '2024-01-01 01:00', '2024-01-01 01:00'], name='datetime')
        return StaticDataset({'flag': pd.DataFrame({'a': [1, 2, 3]}, index=index)}, name=name)

    def test_default_keeps_first_row_with_warning(self, caplog):
        with caplog.at_level(logging.WARNING):
            result = self.duplicated_dataset(name='scenario').fetch('flag')
        assert result['a'].tolist() == [1, 2]
        assert 'Dropped 1 rows' in caplog.text
        assert 'scenario' in caplog.text

    def test_last_keeps_last_row_with_warning(self, caplog):
        with caplog.at_level(logging.WARNING):
            result = self.duplicated_dataset().fetch('flag', config={'keep_duplicate_indices': 'last'})
        assert result['a'].tolist() == [1, 3]
        assert 'Dropped 1 rows' in caplog.text

    def test_all_keeps_every_row_without_warning(self, caplog):
        with caplog.at_level(logging.WARNING):
            result = self.duplicated_dataset().fetch('flag', config={'keep_duplicate_indices': 'all'})
        assert result['a'].tolist() == [1, 2, 3]
        assert 'Dropped' not in caplog.text

    def test_warning_is_logged_once_per_dataset_class_and_flag(self, caplog):
        with caplog.at_level(logging.WARNING):
            self.duplicated_dataset(name='first').fetch('flag')
            self.duplicated_dataset(name='second').fetch('flag', config={'keep_duplicate_indices': 'last'})
        assert sum('Dropped' in r.getMessage() for r in caplog.records) == 1

    def test_unknown_value_raises(self):
        with pytest.raises(ValueError, match='keep_duplicate_indices'):
            self.duplicated_dataset().fetch('flag', config={'keep_duplicate_indices': 'none'})

    @pytest.mark.parametrize('keep', ['first', 'last', 'all'])
    def test_model_flag_with_duplicate_ids_raises(self, keep):
        """A model flag with duplicate ids raises `ValueError` for every `keep_duplicate_indices`."""
        nodes = pd.DataFrame({'zone': ['DE', 'FR', 'NL']}, index=pd.Index(['n1', 'n2', 'n1'], name='Node'))
        spec = ModelFlagSpec(name='nodes', description='Nodes.', index='Node', topology_type=TopologyTypeEnum.Node)
        dataset = StaticDataset({'nodes': nodes}, specs=(spec,))
        with pytest.raises(ValueError, match=r"model flag nodes has duplicate ids \['n1'\]"):
            dataset.fetch('nodes', config={'keep_duplicate_indices': keep})


class TestIndexSorting:
    @staticmethod
    def dataset(index: pd.Index) -> StaticDataset:
        return StaticDataset({'flag': pd.DataFrame({'a': range(len(index))}, index=index)})

    def test_sorts_a_datetime_index(self):
        index = pd.DatetimeIndex(['2024-01-01 01:00', '2024-01-01 00:00'], name='datetime')
        assert self.dataset(index).fetch('flag')['a'].tolist() == [1, 0]

    def test_sorts_a_non_datetime_time_level(self):
        index = pd.Index([(2, 1), (1, 2), (1, 1)], name='period', tupleize_cols=False)
        result = self.dataset(index).fetch('flag')
        assert list(result.index) == [(1, 1), (1, 2), (2, 1)]

    def test_sorts_a_non_time_index(self):
        index = pd.Index(['b', 'c', 'a'], name='item')
        assert list(self.dataset(index).fetch('flag').index) == ['a', 'b', 'c']

    def test_false_keeps_the_order_of_fetch(self):
        index = pd.Index(['b', 'c', 'a'], name='item')
        result = self.dataset(index).fetch('flag', config={'auto_sort_index': False})
        assert list(result.index) == ['b', 'c', 'a']


class TestUnknownKeys:
    def test_dataset_warns_about_unknown_key(self, caplog):
        ds = make_dataset()
        with caplog.at_level(logging.WARNING):
            ds.fetch('flag', config={'convert_periods': False})
        assert 'convert_periods' in caplog.text

    def test_warning_is_logged_once_per_config_class_and_key(self, caplog):
        ds = make_dataset(name='first')
        other = make_dataset(name='second')
        with caplog.at_level(logging.WARNING):
            ds.fetch('flag', config={'convert_periods': False})
            ds.fetch('flag', config={'convert_periods': True})
            ds.fetch('flag', config={'apply_correction': True})
            other.fetch('flag', config={'convert_periods': False})
        messages = [r.getMessage() for r in caplog.records]
        assert sum('convert_periods' in m for m in messages) == 1
        assert sum('apply_correction' in m for m in messages) == 1

    def test_two_datasets_of_one_class_give_one_warning(self, caplog):
        datasets = [make_dataset(name='first'), make_dataset(name='second')]
        with caplog.at_level(logging.WARNING):
            for ds in datasets:
                ds.fetch('flag', config={'unknown_key': 1})
        assert sum('unknown_key' in r.getMessage() for r in caplog.records) == 1

    def test_key_of_platform_config_does_not_warn_on_interpreters(self, caplog):
        class ConfiguredPlatform(Platform):
            @classmethod
            def get_config_type(cls):
                return PlatformConfig

        @ConfiguredPlatform.register_interpreter
        class PriceInterpreter(Interpreter):
            FLAGS = frozenset({'price'})

            def _fetch(self, flag, effective_config, **kwargs):
                return self.prices

        @ConfiguredPlatform.register_interpreter
        class MeanPriceInterpreter(Interpreter):
            FLAGS = frozenset({'mean_price'})

            def _fetch(self, flag, effective_config, **kwargs):
                return self.parent_dataset.fetch('price', effective_config).mean(axis=1).to_frame('mean')

        ds = ConfiguredPlatform(name='scenario', prices=hourly_frame())
        with caplog.at_level(logging.WARNING):
            ds.fetch('price', config={'convert_periods': False})
            ds.fetch('mean_price', config={'convert_periods': False})
        assert 'convert_periods' not in caplog.text

    def test_collection_passes_unknown_key_to_children(self, caplog):
        child = make_dataset(PlatformStaticDataset, name='child')
        collection = DatasetLinkCollection([child], name='collection')
        with caplog.at_level(logging.WARNING):
            collection.fetch('flag', config={'convert_periods': False})
        assert child.seen_configs[-1].convert_periods is False
        assert 'convert_periods' not in caplog.text


class TestCollectionPassDown:
    def test_child_class_config_survives_fetch_through_collection(self):
        DatasetConfigManager.set_class_config(
            PlatformStaticDataset, PlatformConfig(convert_periods=False, auto_sort_index=False)
        )
        child = make_dataset(PlatformStaticDataset, name='child')
        collection = DatasetLinkCollection([child], name='collection')
        collection.fetch('flag')
        assert child.seen_configs[-1].convert_periods is False
        assert child.seen_configs[-1].auto_sort_index is False

    def test_clear_passes_through_collection_to_child_default(self):
        DatasetConfigManager.set_class_config(NoCacheDataset, NoCacheConfig(use_cache=True))
        child = make_dataset(NoCacheDataset, name='child')
        collection = DatasetLinkCollection([child], name='collection')
        collection.fetch('flag', config={'use_cache': DatasetConfig.CLEAR})
        assert child.seen_configs[-1].use_cache is False

    def test_value_after_clear_is_passed_as_value(self):
        cleared = DatasetConfig().merge({'use_cache': DatasetConfig.CLEAR})
        reset = cleared.merge({'use_cache': False})
        assert NoCacheConfig().merge(reset).use_cache is False
        cleared.use_cache = True
        assert NoCacheConfig().merge(cleared).use_cache is True

    def test_explicit_collection_field_overrides_child(self):
        child = make_dataset(PlatformStaticDataset, name='child', config=PlatformConfig(use_cache=True))
        collection = DatasetLinkCollection([child], name='collection', config=DatasetConfig(use_cache=False))
        collection.fetch('flag')
        assert child.seen_configs[-1].use_cache is False


class TestGlobalState:
    def test_clear_class_configs(self):
        DatasetConfigManager.set_class_config(StaticDataset, DatasetConfig(use_cache=False))
        DatasetConfigManager.clear_class_configs()
        assert make_dataset().instance_config.use_cache is True

    def test_update_class_config_kwargs_without_class_config_raises(self):
        with pytest.raises(KeyError, match='set_class_config'):
            DatasetConfigManager.update_class_config_kwargs(StaticDataset, use_cache=False)

    def test_update_class_config_kwargs(self):
        DatasetConfigManager.set_class_config(StaticDataset, DatasetConfig())
        DatasetConfigManager.update_class_config_kwargs(StaticDataset, use_cache=False)
        assert make_dataset().instance_config.use_cache is False


class TestInstanceConfigKwargs:
    def test_set_instance_config_kwargs_without_instance_config(self):
        ds = make_dataset(PlatformStaticDataset)
        ds.set_instance_config_kwargs(convert_periods=False)
        assert type(ds._config) is PlatformConfig
        assert ds.instance_config.convert_periods is False
        assert ds._config.explicit_fields == {'convert_periods'}


class TestDescriptions:
    def test_every_base_field_has_a_description(self):
        for field in dataclasses.fields(DatasetConfig):
            assert field.metadata.get('description'), field.name

    def test_describe_config(self):
        DatasetConfigManager.set_class_config(PlatformStaticDataset, PlatformConfig(convert_periods=False))
        ds = make_dataset(PlatformStaticDataset, config=PlatformConfig(use_cache=False))
        table = ds.describe_config()
        assert isinstance(table, pd.DataFrame)
        assert set(table.index) == {f.name for f in dataclasses.fields(PlatformConfig)}
        assert list(table.columns) == ['type', 'default', 'value', 'layer', 'description']
        assert table.loc['use_cache', 'layer'] == 'instance'
        assert not table.loc['use_cache', 'value']
        assert table.loc['use_cache', 'default']
        assert table.loc['convert_periods', 'layer'] == 'class PlatformStaticDataset'
        assert table.loc['auto_sort_index', 'layer'] == 'default'
        assert table.loc['convert_periods', 'description'] == 'Convert period numbers to timestamps.'

    def test_describe_config_names_layer_of_explicit_none(self):
        DatasetConfigManager.set_class_config(StaticDataset, DatasetConfig(time_index_filter=FIRST_DAY))
        ds = make_dataset(config=DatasetConfig(time_index_filter=None))
        assert ds.describe_config().loc['time_index_filter', 'layer'] == 'instance'
