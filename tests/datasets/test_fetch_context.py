import copy
import dataclasses
import pickle

import pandas as pd
import pytest

from mesqual.datasets import (
    Dataset,
    DatasetComparison,
    DatasetConcatCollection,
    DatasetConfig,
    DatasetLinkCollection,
    DatasetMergeCollection,
    DatasetSumCollection,
    TimeIndexFilter,
)
from mesqual.datasets.dataset_comparison import DatasetConcatCollectionOfComparisons
from mesqual.datasets import INHERIT, FetchRecord
from mesqual.datasets.fetch_context import FetchContext
from mesqual.enums import ComparisonTypeEnum, TopologyTypeEnum
from mesqual.flag import Dimension, ModelFlagSpec, TimeSeriesFlagSpec
from mesqual.units import Units
from tests.datasets.fixtures import (
    Interpreter,
    PlatformConfig,
    PlatformStaticDataset,
    StaticDataset,
    hourly_frame,
    series_spec,
)
from tests.datasets.test_platform_dataset import (
    MEAN_PRICE,
    PRICE,
    VOLUME,
    NodePriceInterpreter,
    PriceCapInterpreter,
    platform_classes,
    prices,
    study_with_price_cap,
)

A, B, C = 'a', 'b', 'c'


def tree(record) -> list:
    """The dependency tree of a record as nested `(class name, flag, children)` tuples."""
    return [(d.dataset_class.rsplit('.', 1)[-1], d.flag, tree(d)) for d in record.dependencies]


class ChainDataset(Dataset):
    """`a` fetches `b`, `b` fetches `c`; `c` reads `convert_periods`."""

    flag_specs = (series_spec(A), series_spec(B), series_spec(C))

    @classmethod
    def get_config_type(cls):
        return PlatformConfig

    def _fetch(self, flag, effective_config, **params):
        if flag == A:
            return self.fetch(B) + 1
        if flag == B:
            return self.fetch(C) + 1
        _ = effective_config.convert_periods
        return hourly_frame(periods=2)


class TestRecords:
    def test_record_holds_call_and_version(self):
        ds = ChainDataset(name='ds')
        ds.fetch(C, config={'use_cache': False})
        record = ds.get_fetch_record(C, config={'use_cache': False})
        assert record.dataset_class == 'tests.datasets.test_fetch_context.ChainDataset'
        assert record.dataset_name == 'ds'
        assert record.flag == C
        assert record.call_config == {'use_cache': False}
        assert record.version == (('ChainDataset', 1),)

    def test_dependencies_form_a_tree(self):
        ds = ChainDataset()
        ds.fetch(A)
        record = ds.get_fetch_record(A)
        assert tree(record) == [('ChainDataset', B, [('ChainDataset', C, [])])]
        assert [d.flag for d in record.iter_dependencies()] == [B, C]
        assert ds.get_fetch_record(C).dependencies == ()

    def test_one_flag_under_two_parents_appears_under_each(self):
        class TwoParents(ChainDataset):
            def _fetch(self, flag, effective_config, **params):
                if flag == A:
                    return self.fetch(B, {'convert_periods': True}) + self.fetch(B, {'convert_periods': False})
                if flag == B:
                    return self.fetch(C, effective_config)
                return super()._fetch(flag, effective_config, **params)

        ds = TwoParents()
        ds.fetch(A)
        first, second = ds.get_fetch_record(A).dependencies
        assert [(d.flag, d.config_reads['convert_periods']) for d in first.dependencies] == [(C, True)]
        assert [(d.flag, d.config_reads['convert_periods']) for d in second.dependencies] == [(C, False)]
        assert [d.flag for d in ds.get_fetch_record(A).iter_dependencies()] == [B, C, B, C]

    def test_config_reads_of_fetch_and_post_processing_are_recorded(self):
        ds = ChainDataset()
        ds.fetch(C)
        reads = ds.get_fetch_record(C).config_reads
        assert reads['convert_periods'] is True
        assert 'auto_sort_index' in reads
        assert 'use_cache' not in reads
        assert 'param_defaults' not in reads

    def test_reading_the_whole_config_records_all_fields(self):
        for read_all in (dataclasses.asdict, vars):

            class ReadAll(Dataset):
                flag_specs = (series_spec(C),)

                def _fetch(self, flag, effective_config, **params):
                    read_all(effective_config)
                    return hourly_frame(periods=2)

            ds = ReadAll()
            ds.fetch(C)
            assert set(ds.get_fetch_record(C).config_reads) == DatasetConfig._field_names()

    def test_no_record_outside_a_fetch(self):
        config = DatasetConfig()
        assert config.use_cache is True
        assert FetchContext.current() is None


class TestCallKeys:
    def test_two_fetches_of_one_flag_with_different_config_are_two_dependencies(self):
        class Two(Dataset):
            flag_specs = (series_spec(A), series_spec(B))

            @classmethod
            def get_config_type(cls):
                return PlatformConfig

            def _fetch(self, flag, effective_config, **params):
                if flag == A:
                    return self.fetch(B, config={'convert_periods': True}) + self.fetch(
                        B, config={'convert_periods': False}
                    )
                _ = effective_config.convert_periods
                return hourly_frame(periods=2)

        ds = Two()
        ds.fetch(A)
        dependencies = ds.get_fetch_record(A).dependencies
        assert [(d.call_config, d.config_reads['convert_periods']) for d in dependencies] == [
            ({'convert_periods': True}, True),
            ({'convert_periods': False}, False),
        ]

    def test_nested_dependency_holds_call_kwargs_config_and_params(self):
        inner = StaticDataset(
            {'p': hourly_frame(periods=2)},
            name='inner',
            specs=(series_spec('p', params=(Dimension('scenario', 'Scenario.', {'low': 'low', 'high': 'high'}),)),),
        )
        comparison = DatasetComparison(
            variation_dataset=StaticDataset({'x': hourly_frame(periods=2)}, name='var'),
            reference_dataset=StaticDataset({'x': hourly_frame(periods=2)}, name='ref'),
        )

        class Outer(Dataset):
            flag_specs = (series_spec(A),)

            def _fetch(self, flag, effective_config, **params):
                inner.fetch('p', config={'use_cache': False}, scenario='low')
                comparison.fetch('x', comparison_type=ComparisonTypeEnum.BOTH)
                return hourly_frame(periods=2)

        ds = Outer()
        ds.fetch(A)
        p, x = ds.get_fetch_record(A).dependencies
        assert (p.call_kwargs, p.call_config, p.params) == (
            {'scenario': 'low'},
            {'use_cache': False},
            {'scenario': 'low'},
        )
        assert x.call_kwargs == {'comparison_type': ComparisonTypeEnum.BOTH}
        assert x.params['comparison_type'] is ComparisonTypeEnum.BOTH
        assert [(d.dataset_name, d.call_config) for d in x.dependencies] == [('var', INHERIT), ('ref', INHERIT)]

    def test_call_kwargs_are_part_of_the_key(self):
        solution = Dimension('solution', 'Solutions.', {'final': 'final', 'first': 'first'}, default='final')

        class Explicit(Dataset):
            flag_specs = (series_spec(A), series_spec(B, params=(solution,)))

            def _fetch(self, flag, effective_config, **params):
                if flag == A:
                    return self.fetch(B, effective_config, solution='first') + self.fetch(B, effective_config)
                return hourly_frame(periods=2)

        ds = Explicit()
        config = {'param_defaults': {'solution': 'first'}}
        ds.fetch(A, config=config)
        dependencies = ds.get_fetch_record(A, config=config).dependencies
        assert [(d.call_kwargs, d.params) for d in dependencies] == [
            ({'solution': 'first'}, {'solution': 'first'}),
            ({}, {'solution': 'first'}),
        ]
        assert dependencies[1].param_default_reads == {'solution': 'first'}
        assert dependencies[0].param_default_reads == {}

    def test_get_fetch_record_finds_the_call_with_this_config(self):
        ds = ChainDataset()
        ds.fetch(C, config={'convert_periods': True})
        ds.fetch(C, config={'convert_periods': False})
        record = ds.get_fetch_record(C, config={'convert_periods': True})
        assert record.call_config == {'convert_periods': True}
        assert record.config_reads['convert_periods'] is True


class TestDatasetParams:
    @staticmethod
    def comparison(**kwargs) -> DatasetComparison:
        return DatasetComparison(
            variation_dataset=StaticDataset({'x': hourly_frame(periods=2) * 2}, name='var'),
            reference_dataset=StaticDataset({'x': hourly_frame(periods=2)}, name='ref'),
            **kwargs,
        )

    def test_comparison_params_key_the_records(self):
        comparison = self.comparison()
        comparison.fetch('x', comparison_type=ComparisonTypeEnum.BOTH)
        comparison.fetch('x')
        both = comparison.get_fetch_record('x', comparison_type=ComparisonTypeEnum.BOTH)
        delta = comparison.get_fetch_record('x')
        assert both.params['comparison_type'] is ComparisonTypeEnum.BOTH
        assert delta.params == {
            'comparison_type': ComparisonTypeEnum.DELTA,
            'replace_unchanged_values_by_nan': False,
            'fill_value': None,
        }
        assert delta.call_kwargs == {}

    def test_concat_axis_is_no_fetch_param(self):
        collection = DatasetConcatCollection([StaticDataset({'x': hourly_frame(periods=2)}, name='a')])
        assert DatasetConcatCollection.fetch_params == DatasetConcatCollectionOfComparisons.fetch_params == ()
        with pytest.raises(ValueError, match=r"declares the parameters \['concat_axis'\]"):
            collection.fetch('x', concat_axis=0)

    def test_explicit_default_and_resolved_default_give_one_resolved_params(self):
        comparison = self.comparison()
        comparison.fetch('x')
        comparison.fetch('x', comparison_type=ComparisonTypeEnum.DELTA)
        implicit = comparison.get_fetch_record('x')
        explicit = comparison.get_fetch_record('x', comparison_type=ComparisonTypeEnum.DELTA)
        assert implicit.params == explicit.params
        assert implicit.key != explicit.key

    def test_dataset_param_resolves_through_param_defaults(self):
        comparison = self.comparison(config=DatasetConfig(param_defaults={'comparison_type': ComparisonTypeEnum.BOTH}))
        expected = self.comparison().fetch('x', comparison_type=ComparisonTypeEnum.BOTH)
        pd.testing.assert_frame_equal(comparison.fetch('x'), expected)
        record = comparison.get_fetch_record('x')
        assert record.call_kwargs == {}
        assert record.params['comparison_type'] is ComparisonTypeEnum.BOTH
        assert record.param_default_reads == {'comparison_type': ComparisonTypeEnum.BOTH}

    def test_call_wins_over_param_defaults_for_a_dataset_param(self):
        comparison = self.comparison(config=DatasetConfig(param_defaults={'comparison_type': ComparisonTypeEnum.BOTH}))
        comparison.fetch('x', comparison_type=ComparisonTypeEnum.VARIATION)
        record = comparison.get_fetch_record('x', comparison_type=ComparisonTypeEnum.VARIATION)
        assert record.params['comparison_type'] is ComparisonTypeEnum.VARIATION
        assert record.param_default_reads == {}

    def test_dataset_param_outside_its_values_raises(self):
        with pytest.raises(ValueError, match="is not a value of 'comparison_type'"):
            self.comparison().fetch('x', comparison_type='delta')

    def test_record_of_a_call_with_flag_and_dataset_params(self):
        solution = Dimension('solution', 'Solutions.', {'final': 'final', 'first': 'first'}, default='final')

        class Prices(Dataset):
            flag_specs = (series_spec('price', params=(solution,)),)
            fetch_params = (Dimension('currency', 'Currency.', {'EUR': 'euro', 'USD': 'dollar'}, default='EUR'),)

            def _fetch(self, flag, effective_config, currency, solution):
                return hourly_frame(periods=2) * (2 if currency == 'USD' else 1)

        ds = Prices(config=DatasetConfig(param_defaults={'solution': 'first'}))
        ds.fetch('price', config={'use_cache': False}, currency='USD')
        record = ds.get_fetch_record('price', config={'use_cache': False}, currency='USD')
        assert record.call_kwargs == {'currency': 'USD'}
        assert record.call_config == {'use_cache': False}
        assert record.params == {'solution': 'first', 'currency': 'USD'}
        assert record.param_default_reads == {'solution': 'first'}


class TestParamNameClash:
    SOLUTION = Dimension('solution', 'Solutions.', {'final': 'final'}, default='final')

    def test_dataset_with_flag_and_dataset_param_of_one_name_raises(self):
        class Clashing(Dataset):
            flag_specs = (series_spec('x', params=(TestParamNameClash.SOLUTION,)),)
            fetch_params = (TestParamNameClash.SOLUTION,)

            def _fetch(self, flag, effective_config, **params):
                return hourly_frame(periods=2)

        with pytest.raises(ValueError, match=r"\['solution'\] are flag parameters .* and dataset parameters"):
            Clashing()
        with pytest.raises(ValueError, match=r"\['solution'\] are flag parameters .* and dataset parameters"):
            Clashing(parent_dataset=object())

    def test_dataset_with_a_parent_checks_its_own_specs_only(self):
        class WithParam(Dataset):
            flag_specs = (series_spec('x', params=(TestParamNameClash.SOLUTION,)),)

            def _fetch(self, flag, effective_config, **params):
                return hourly_frame(periods=2)

        class TakesSolution(WithParam):
            flag_specs = ()
            fetch_params = (TestParamNameClash.SOLUTION,)

        WithParam(parent_dataset=object())
        TakesSolution(parent_dataset=WithParam())

    def test_collection_of_a_flag_param_and_a_dataset_param_of_one_name_raises(self):
        plain = StaticDataset(
            {'x': hourly_frame(periods=2)},
            name='plain',
            specs=(series_spec('x', params=(Dimension('comparison_type', 'Clash.', None, default=None),)),),
        )
        comparison = TestDatasetParams.comparison()
        with pytest.raises(ValueError, match=r"\['comparison_type'\] are flag parameters"):
            DatasetMergeCollection([plain, comparison])
        collection = DatasetMergeCollection([comparison])
        with pytest.raises(ValueError, match=r"\['comparison_type'\] are flag parameters"):
            collection.add_dataset(plain)
        assert collection.datasets == [comparison]


class TestInheritedConfig:
    def test_child_record_inherits_the_collection_config(self):
        class Reader(PlatformStaticDataset):
            def _fetch(self, flag, effective_config, **kwargs):
                _ = effective_config.convert_periods
                return super()._fetch(flag, effective_config, **kwargs)

        class Collection(DatasetMergeCollection):
            @classmethod
            def get_config_type(cls):
                return PlatformConfig

        collection = Collection(
            [Reader({'x': hourly_frame(periods=2)}, name='c')], config=PlatformConfig(convert_periods=False)
        )
        collection.fetch('x')
        child = collection.get_fetch_record('x').dependencies[0]
        assert child.call_config is INHERIT
        assert child.config_reads['convert_periods'] is False
        assert 'convert_periods' not in collection.get_fetch_record('x').config_reads

        collection.set_instance_config_kwargs(convert_periods=True)
        collection.fetch('x')
        child_after = collection.get_fetch_record('x').dependencies[0]
        assert child_after.call_config is INHERIT
        assert child_after.config_reads['convert_periods'] is True
        assert child_after.key == child.key

    def test_derived_flag_passing_its_config_inherits(self):
        class Derived(ChainDataset):
            def _fetch(self, flag, effective_config, **params):
                if flag == A:
                    return self.fetch(C, effective_config)
                return super()._fetch(flag, effective_config, **params)

        ds = Derived()
        ds.fetch(A)
        assert ds.get_fetch_record(A).dependencies[0].call_config is INHERIT
        assert ds.get_fetch_record(C, config=INHERIT) is ds.get_fetch_record(A).dependencies[0]


class TestPlatformRecords:
    def test_routed_calls_have_their_own_records(self):
        platform, _ = platform_classes()
        ds = platform(prices=prices())
        ds.fetch(MEAN_PRICE)
        record = ds.get_fetch_record(MEAN_PRICE)
        assert tree(record) == [
            ('MeanPriceInterpreter', MEAN_PRICE, [('MyPlatformDataset', PRICE, [('NodePriceInterpreter', PRICE, [])])]),
        ]
        assert record.version == (('MeanPriceInterpreter', 1),)

    def test_direct_sibling_fetch_is_a_dependency(self):
        platform, _ = platform_classes()

        class Ratio(Interpreter):
            FLAGS = frozenset({'ratio'})

            def _fetch(self, flag, effective_config, **kwargs):
                return self.parent_dataset.get_interpreter_instance(NodePriceInterpreter).fetch(VOLUME)

        platform.register_interpreter(Ratio)
        ds = platform(prices=prices())
        ds.fetch('ratio')
        assert tree(ds.get_fetch_record('ratio')) == [('Ratio', 'ratio', [('NodePriceInterpreter', VOLUME, [])])]

    def test_override_call_is_the_only_dependency(self):
        _, study = study_with_price_cap()
        ds = study(prices=prices())
        ds.fetch(PRICE)
        assert tree(ds.get_fetch_record(PRICE)) == [('PriceCapInterpreter', PRICE, [])]


class TestLinkCollectionRecords:
    @staticmethod
    def link(wired: bool, child) -> DatasetLinkCollection:
        link = DatasetLinkCollection([child], name='link')
        if wired:
            child.parent_dataset = link
        return link

    @pytest.mark.parametrize('wired', [False, True])
    def test_child_call_is_a_dependency(self, wired):
        link = self.link(wired, StaticDataset({'x': hourly_frame(periods=2)}, name='a'))
        link.fetch('x')
        assert tree(link.get_fetch_record('x')) == [('StaticDataset', 'x', [])]

    @pytest.mark.parametrize('wired', [False, True])
    def test_child_instance_config_read_stays_in_the_child_record(self, wired):
        child = ChainDataset(name='child', config=PlatformConfig(convert_periods=False))
        link = self.link(wired, child)
        link.fetch(C)
        record = link.get_fetch_record(C)
        assert 'convert_periods' not in record.config_reads
        assert record.dependencies[0].config_reads['convert_periods'] is False

    @pytest.mark.parametrize('wired', [False, True])
    def test_param_defaults_through_concat_and_link_give_different_records(self, wired):
        solution = Dimension('solution', 'Solutions.', {'final': 'final', 'first': 'first'}, default='final')
        child = StaticDataset(
            {'p': hourly_frame(periods=2)}, name='child', specs=(series_spec('p', params=(solution,)),)
        )
        concat = DatasetConcatCollection([self.link(wired, child)], name='concat')
        leaves = []
        for value in ('first', 'final'):
            concat.fetch('p', config={'param_defaults': {'solution': value}})
            [link_record] = concat.get_fetch_record('p', config={'param_defaults': {'solution': value}}).dependencies
            [leaf] = link_record.dependencies
            leaves.append((leaf.params, leaf.param_default_reads))
        assert leaves == [
            ({'solution': 'first'}, {'solution': 'first'}),
            ({'solution': 'final'}, {'solution': 'final'}),
        ]


class TestWatchedConfig:
    def test_copy_and_pickle_record_all_fields_and_drop_the_recorder(self):
        for copy_config in (copy.copy, copy.deepcopy, lambda c: pickle.loads(pickle.dumps(c))):
            reads = []
            watched = DatasetConfig()._watched(lambda name, value: reads.append(name))
            with FetchContext.frame(FetchRecord('x', 'x', 'x', {}, {}, {}, ())):
                copied = copy_config(watched)
                assert set(reads) == DatasetConfig._field_names()
                reads.clear()
                _ = copied.use_cache
            assert reads == []

    def test_copy_inside_fetch_records_all_fields(self):
        class CopyReader(Dataset):
            flag_specs = (series_spec(C),)

            @classmethod
            def get_config_type(cls):
                return PlatformConfig

            def _fetch(self, flag, effective_config, **params):
                _ = copy.copy(effective_config).convert_periods
                return hourly_frame(periods=2)

        ds = CopyReader()
        ds.fetch(C)
        assert set(ds.get_fetch_record(C).config_reads) == PlatformConfig._field_names()


class TestTimeIndexFilterScope:
    def test_filter_skips_model_flags(self):
        model_spec = ModelFlagSpec(
            name='Node.Model', description='Nodes.', index='Node', topology_type=TopologyTypeEnum.Node
        )

        class ModelWithTimeIndex(Dataset):
            flag_specs = (model_spec,)

            def _fetch(self, flag, effective_config, **params):
                return hourly_frame(periods=48)

        ds = ModelWithTimeIndex(config={'time_index_filter': TimeIndexFilter.dates(include=['2024-01-01'])})
        assert len(ds.fetch('Node.Model')) == 48

    def test_filter_applies_to_time_series_flags(self):
        ds = StaticDataset(
            {C: hourly_frame(periods=48)}, config={'time_index_filter': TimeIndexFilter.dates(include=['2024-01-01'])}
        )
        assert len(ds.fetch(C)) == 24


class TestModelFlagReferences:
    NODE = ModelFlagSpec(name='Node.Model', description='Nodes.', index='Node', topology_type=TopologyTypeEnum.Node)
    PRICE = TimeSeriesFlagSpec(name='Node.price', description='Price.', unit=Units.EUR_per_MWh, model_flag='Node.Model')

    def datasets(self):
        model = StaticDataset(
            {'Node.Model': pd.DataFrame(index=pd.Index(['n1'], name='Node'))}, name='m', specs=(self.NODE,)
        )
        series = StaticDataset({'Node.price': hourly_frame(periods=2)}, name='t', specs=(self.PRICE,))
        return model, series

    def test_series_dataset_with_model_spec_in_a_sibling_works_in_a_link_collection(self):
        model, series = self.datasets()
        assert DatasetLinkCollection([model, series]).fetch('Node.price').shape == (2, 2)

    def test_standalone_dataset_checks_references_in_its_outermost_fetch(self):
        _, series = self.datasets()
        with pytest.raises(ValueError, match='names no ModelFlagSpec'):
            series.fetch('Node.price')

    def test_link_collection_checks_references_at_construction(self):
        _, series = self.datasets()
        with pytest.raises(ValueError, match='names no ModelFlagSpec'):
            DatasetLinkCollection([series])


class TestParamsThroughCollections:
    @staticmethod
    def comparisons() -> list[DatasetComparison]:
        a, b, c = (StaticDataset({'x': hourly_frame(periods=2) * v}, name=n) for v, n in ((1, 'a'), (2, 'b'), (5, 'c')))
        return [DatasetComparison(a, b), DatasetComparison(c, b)]

    def test_collection_accepts_the_dataset_params_of_the_children(self):
        outer = DatasetConcatCollection([DatasetConcatCollectionOfComparisons(self.comparisons(), name='inner')])
        assert set(outer._accepted_dataset_params) == {
            'comparison_type',
            'replace_unchanged_values_by_nan',
            'fill_value',
        }

    def test_plain_concat_of_comparisons_passes_comparison_type(self):
        comparisons = self.comparisons()
        collection = DatasetConcatCollection(comparisons, name='plain')
        expected = pd.concat(
            {c.name: c.fetch('x', comparison_type=ComparisonTypeEnum.BOTH) for c in comparisons},
            axis=1,
            names=['dataset'],
        )
        pd.testing.assert_frame_equal(collection.fetch('x', comparison_type=ComparisonTypeEnum.BOTH), expected)
        record = collection.get_fetch_record('x', comparison_type=ComparisonTypeEnum.BOTH)
        assert record.params == {'comparison_type': ComparisonTypeEnum.BOTH}
        assert all(d.params['comparison_type'] is ComparisonTypeEnum.BOTH for d in record.dependencies)

    def test_concat_of_concat_of_comparisons_passes_comparison_type(self):
        inner = DatasetConcatCollectionOfComparisons(self.comparisons(), name='inner')
        outer = DatasetConcatCollection([inner], name='outer')
        result = outer.fetch('x', comparison_type=ComparisonTypeEnum.BOTH)
        pd.testing.assert_frame_equal(
            result.droplevel(0, axis=1),
            inner.fetch('x', comparison_type=ComparisonTypeEnum.BOTH),
        )
        [inner_record] = outer.get_fetch_record('x', comparison_type=ComparisonTypeEnum.BOTH).dependencies
        assert inner_record.call_kwargs == {'comparison_type': ComparisonTypeEnum.BOTH}
        assert inner_record.params == {'comparison_type': ComparisonTypeEnum.BOTH}
        assert all(d.call_kwargs == {'comparison_type': ComparisonTypeEnum.BOTH} for d in inner_record.dependencies)

    def test_fetch_merged_passes_params_and_keeps_records_on_the_children(self):
        comparisons = self.comparisons()
        collection = DatasetConcatCollection(comparisons, name='plain')
        merged = collection.fetch_merged('x', comparison_type=ComparisonTypeEnum.VARIATION)
        pd.testing.assert_frame_equal(merged, hourly_frame(periods=2))
        record = comparisons[0].get_fetch_record('x', INHERIT, comparison_type=ComparisonTypeEnum.VARIATION)
        assert record is not None


class TestMixedCollections:
    @staticmethod
    def scenarios() -> tuple[StaticDataset, StaticDataset]:
        return StaticDataset({'x': hourly_frame(periods=2)}, name='base'), StaticDataset(
            {'x': hourly_frame(periods=2) * 3}, name='high'
        )

    def test_scen_comp_passes_comparison_params_only_to_comparisons(self):
        from mesqual import StudyManager

        study = StudyManager.factory_from_scenarios(list(self.scenarios()), [('high', 'base')])
        for options in ({'fill_value': 0.0}, {'comparison_type': ComparisonTypeEnum.VARIATION}):
            result = study.scen_comp.fetch('x', **options)
            assert result.shape == (2, 6)
        record = study.scen_comp.get_fetch_record('x', comparison_type=ComparisonTypeEnum.VARIATION)
        by_class = {d.dataset_class.rsplit('.', 1)[-1]: d for d in record.iter_dependencies()}
        assert by_class['DatasetComparison'].params['comparison_type'] is ComparisonTypeEnum.VARIATION
        assert all(d.call_kwargs == {} for d in record.iter_dependencies() if d.dataset_class.endswith('StaticDataset'))

    def test_merge_of_a_comparison_and_a_plain_dataset(self):
        base, high = self.scenarios()
        plain = StaticDataset({'x': hourly_frame(periods=2) * 5}, name='plain')
        collection = DatasetMergeCollection([DatasetComparison(high, base), plain], name='m')
        result = collection.fetch('x', comparison_type=ComparisonTypeEnum.VARIATION)
        pd.testing.assert_frame_equal(result, hourly_frame(periods=2) * 3)
        record = collection.get_fetch_record('x', comparison_type=ComparisonTypeEnum.VARIATION)
        assert record.dependencies[0].call_kwargs == {'comparison_type': ComparisonTypeEnum.VARIATION}
        assert record.dependencies[1].call_kwargs == {}


class TestRecordFields:
    RES = Dimension('res', 'Resolution.', {'hourly': 'h', 'all': 'a'}, default='hourly')
    OTHER = Dimension('other', 'Other.', None, default=0)

    def dataset_type(self, read_whole: bool):
        class ParamDataset(Dataset):
            flag_specs = (series_spec('x', params=(self.RES,)), series_spec('y', params=(self.OTHER,)))

            def _fetch(self, flag, effective_config, **params):
                if read_whole:
                    dataclasses.replace(effective_config)
                return hourly_frame(periods=2)

        return ParamDataset

    def test_param_default_reads_hold_the_entries_used_for_resolution(self):
        ds = self.dataset_type(read_whole=False)(config=DatasetConfig(param_defaults={'res': 'all', 'other': 1}))
        ds.fetch('x')
        record = ds.get_fetch_record('x')
        assert record.param_default_reads == {'res': 'all'}
        assert 'param_defaults' not in record.config_reads

    def test_config_reads_param_defaults_is_the_field_value(self):
        ds = self.dataset_type(read_whole=True)(config=DatasetConfig(param_defaults={'res': 'all', 'other': 1}))
        ds.fetch('x')
        record = ds.get_fetch_record('x')
        assert record.config_reads['param_defaults'] == {'res': 'all', 'other': 1}
        assert record.param_default_reads == {'res': 'all'}

    def test_recorded_values_are_snapshots(self):
        @dataclasses.dataclass
        class WeightsConfig(DatasetConfig):
            weights: dict = None

        class Weighted(Dataset):
            flag_specs = (series_spec('x'), series_spec('y'))

            @classmethod
            def get_config_type(cls):
                return WeightsConfig

            def _fetch(self, flag, effective_config, **params):
                _ = effective_config.weights
                return hourly_frame(periods=2)

        instance_weights, call_weights = {'a': 1}, {'b': 1}
        ds = Weighted(config=WeightsConfig(weights=instance_weights))
        ds.fetch('x')
        ds.fetch('y', config={'weights': call_weights})
        instance_weights['a'] = 2
        call_weights['b'] = 2
        assert ds.get_fetch_record('x').config_reads['weights'] == {'a': 1}
        [y_record] = [r for r in ds._fetch_records.values() if r.flag == 'y']
        assert y_record.call_config == {'weights': {'b': 1}}
        assert y_record.config_reads['weights'] == {'b': 1}

    def test_record_of_a_call_that_raised(self):
        class Failing(Dataset):
            flag_specs = (series_spec(A), series_spec(B))

            def _fetch(self, flag, effective_config, **params):
                if flag == B:
                    raise RuntimeError('no data')
                try:
                    self.fetch(B)
                except RuntimeError:
                    pass
                return hourly_frame(periods=2)

        ds = Failing()
        ds.fetch(A)
        record = ds.get_fetch_record(A)
        assert record.raised is False
        [dependency] = record.dependencies
        assert dependency.raised is True
        with pytest.raises(RuntimeError):
            ds.fetch(B)
        assert ds.get_fetch_record(B).raised is True

    def test_records_hash_by_key(self):
        ds = ChainDataset()
        ds.fetch(A)
        record = ds.get_fetch_record(A)
        copied = dataclasses.replace(record)
        assert hash(record) == hash(copied) == hash(record.key)
        assert len({record, copied, *record.iter_dependencies()}) == 3

    def test_sum_collection_merges_model_flags_without_a_temporary_collection(self):
        node = ModelFlagSpec(name='N.Model', description='Nodes.', index='Node', topology_type=TopologyTypeEnum.Node)
        frame = pd.DataFrame({'x': [1]}, index=pd.Index(['n1'], name='Node'))
        children = [StaticDataset({'N.Model': frame}, name=n, specs=(node,)) for n in ('m1', 'm2')]
        collection = DatasetSumCollection(children, name='sum')
        collection.fetch('N.Model')
        assert tree(collection.get_fetch_record('N.Model')) == [
            ('StaticDataset', 'N.Model', []),
            ('StaticDataset', 'N.Model', []),
        ]


class TestReadOnlyWatchedConfig:
    def test_setting_a_field_of_the_effective_config_raises(self):
        class Mutating(Dataset):
            flag_specs = (series_spec(A),)

            @classmethod
            def get_config_type(cls):
                return PlatformConfig

            def _fetch(self, flag, effective_config, **params):
                effective_config.convert_periods = False

        with pytest.raises(AttributeError, match=r'config\.merge'):
            Mutating().fetch(A)

    def test_merge_passes_a_changed_config_on(self):
        class Merging(Dataset):
            flag_specs = (series_spec(A), series_spec(B))

            @classmethod
            def get_config_type(cls):
                return PlatformConfig

            def _fetch(self, flag, effective_config, **params):
                if flag == A:
                    return self.fetch(B, effective_config.merge({'convert_periods': False}))
                _ = effective_config.convert_periods
                return hourly_frame(periods=2)

        ds = Merging()
        ds.fetch(A)
        record = ds.get_fetch_record(A)
        assert set(record.config_reads) >= PlatformConfig._field_names()
        [dependency] = record.dependencies
        assert dependency.call_config['convert_periods'] is False
        assert dependency.config_reads['convert_periods'] is False

    def test_repr_records_all_fields(self):
        reads = []
        watched = DatasetConfig()._watched(lambda name, value: reads.append(name))
        with FetchContext.frame(FetchRecord('x', 'x', 'x', {}, {}, {}, ())):
            repr(watched)
        assert set(reads) == DatasetConfig._field_names()


class TestInterpreterRecords:
    @staticmethod
    def record_of_direct_override_fetch(cap_version: int):
        _, study = platform_classes()

        class Cap(NodePriceInterpreter):
            def _fetch(self, flag, effective_config, **kwargs):
                return super()._fetch(flag, effective_config, **kwargs).clip(upper=1000 * cap_version)

        study.register_interpreter(Cap, override_interpreter=NodePriceInterpreter, version=cap_version)
        ds = study(name='st', prices=prices())
        cap = ds.get_interpreter_instance(Cap)
        cap.fetch(PRICE)
        return cap.get_fetch_record(PRICE)

    def test_override_version_bump_changes_the_record_of_the_override(self):
        cap_1 = self.record_of_direct_override_fetch(1)
        cap_2 = self.record_of_direct_override_fetch(2)
        qualname = 'TestInterpreterRecords.record_of_direct_override_fetch.<locals>.Cap'
        assert cap_1.version == (('NodePriceInterpreter', 1), (qualname, 1))
        assert cap_2.version == (('NodePriceInterpreter', 1), (qualname, 2))
        assert cap_1.key == cap_2.key and cap_1 != cap_2

    def test_interpreter_names_are_stable(self):
        _, study = study_with_price_cap()
        records = []
        for _ in range(2):
            ds = study(name='st', prices=prices())
            assert ds.get_interpreter_instance(PriceCapInterpreter).name == 'st.PriceCapInterpreter'
            ds.fetch(PRICE)
            record = ds.get_fetch_record(PRICE)
            records.append([record.key] + [d.key for d in record.iter_dependencies()])
        assert records[0] == records[1]
