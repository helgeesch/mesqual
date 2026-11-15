import inspect

import pandas as pd
import pytest

from mesqual.datasets import (
    UNSET,
    Dataset,
    DatasetCollection,
    Unset,
    DatasetComparison,
    DatasetConcatCollection,
    DatasetConcatCollectionOfComparisons,
    DatasetConfig,
)
from mesqual.enums import ComparisonTypeEnum, ItemTypeEnum
from mesqual.units import Units
from mesqual.flag import Dimension, TimeSeriesFlagSpec
from tests.datasets.fixtures import StaticDataset, hourly_frame, series_spec

PRICE = 'Node.Results.price'
LOAD = 'Node.Results.load'
SOLUTION = Dimension('solution', 'Solutions.', {'final': 'final', 'first': 'first'}, default='final')
SCENARIO = Dimension('scenario', 'Scenario.', {'low': 'low', 'high': 'high'})


def positive(value):
    if value <= 0:
        raise ValueError('threshold must be positive')


THRESHOLD = Dimension('threshold', 'Threshold.', default=1, validate=positive)


class ParamDataset(Dataset):
    """Returns a frame and stores the parameters that reach `_fetch`."""

    flag_specs = (series_spec(PRICE, params=(SOLUTION, SCENARIO, THRESHOLD)), series_spec(LOAD))

    def __init__(self, name: str = None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.seen_params: list[dict] = []

    def _fetch(self, flag, effective_config, **params):
        self.seen_params.append(params)
        return hourly_frame(periods=2)


class TestResolution:
    def test_fetch_argument_wins_over_param_defaults_and_default(self):
        ds = ParamDataset(config={'param_defaults': {'solution': 'first', 'scenario': 'low'}})
        ds.fetch(PRICE, solution='final', scenario='high')
        assert ds.seen_params[-1] == {'solution': 'final', 'scenario': 'high', 'threshold': 1}

    def test_param_defaults_win_over_dimension_default(self):
        ds = ParamDataset(config={'param_defaults': {'solution': 'first', 'scenario': 'low'}})
        ds.fetch(PRICE)
        assert ds.seen_params[-1] == {'solution': 'first', 'scenario': 'low', 'threshold': 1}

    def test_param_defaults_at_fetch_time_merge_per_key(self):
        ds = ParamDataset(config={'param_defaults': {'solution': 'first'}})
        ds.fetch(PRICE, config={'param_defaults': {'scenario': 'low'}})
        assert ds.seen_params[-1]['solution'] == 'first'

    def test_flag_without_params_gets_none(self):
        ds = ParamDataset()
        ds.fetch(LOAD)
        assert ds.seen_params[-1] == {}


class TestErrors:
    def test_missing_required_value_raises(self):
        with pytest.raises(
            ValueError,
            match=(f"^Flag {PRICE}: parameter 'scenario' is required. Pass it to fetch or set param_defaults.$"),
        ):
            ParamDataset().fetch(PRICE)

    def test_invalid_value_in_param_defaults_raises_with_flag_prefix(self):
        ds = ParamDataset(config={'param_defaults': {'scenario': 'mid'}})
        with pytest.raises(
            ValueError, match=(f"^Flag {PRICE}: 'mid' is not a value of 'scenario'. Allowed: \\['low', 'high'\\].$")
        ):
            ds.fetch(PRICE)

    def test_unknown_parameter_raises(self):
        with pytest.raises(ValueError, match='no flag spec or dataset class declares the parameters'):
            ParamDataset().fetch(PRICE, scenario='low', unknown='x')

    def test_value_outside_values_raises(self):
        with pytest.raises(ValueError, match='not a value of'):
            ParamDataset().fetch(PRICE, scenario='mid')

    def test_failing_validate_raises(self):
        with pytest.raises(ValueError, match='positive'):
            ParamDataset().fetch(PRICE, scenario='low', threshold=-1)

    def test_param_defaults_name_that_no_spec_declares_raises(self):
        ds = ParamDataset(config={'param_defaults': {'unknown': 'x'}})
        with pytest.raises(ValueError, match='param_defaults names parameters that this dataset does not declare'):
            ds.fetch(LOAD)


class OtherDataset(Dataset):
    flag_specs = (series_spec(LOAD),)

    def _fetch(self, flag, effective_config, **params):
        assert params == {}
        return hourly_frame(periods=2)


class TestCollections:
    def test_concat_passes_param_defaults_to_children_that_do_not_declare_them(self):
        collection = DatasetConcatCollection([ParamDataset('a'), OtherDataset(name='b')])
        config = {'param_defaults': {'scenario': 'low'}}
        result = collection.fetch(LOAD, config=config)
        assert set(result.columns.get_level_values(0)) == {'a', 'b'}
        collection.fetch(PRICE, config=config)
        assert collection.datasets[0].seen_params[-1]['scenario'] == 'low'

    def test_concat_raises_for_name_that_no_child_declares(self):
        collection = DatasetConcatCollection([ParamDataset('a'), OtherDataset(name='b')])
        with pytest.raises(ValueError, match='param_defaults names parameters that this dataset does not declare'):
            collection.fetch(LOAD, config={'param_defaults': {'unknown': 'x'}})

    def test_collection_passes_each_child_only_its_dataset_params(self):
        class Tagged(OtherDataset):
            fetch_params = (Dimension('tag', 'Tag.', None, default='none'),)

            def __init__(self, name: str):
                super().__init__(name=name)
                self.seen_params: list[dict] = []

            def _fetch(self, flag, effective_config, **params):
                self.seen_params.append(params)
                return hourly_frame(periods=2)

        class Weighted(Tagged):
            fetch_params = (Dimension('weight', 'Weight.', None, default=1),)

        tagged, weighted, plain = Tagged('t'), Weighted('w'), OtherDataset(name='p')
        collection = DatasetConcatCollection([tagged, weighted, plain])
        collection.fetch(LOAD, tag='x', weight=2)
        assert tagged.seen_params[-1] == {'tag': 'x'}
        assert weighted.seen_params[-1] == {'tag': 'x', 'weight': 2}
        assert collection.get_fetch_record(LOAD, tag='x', weight=2).params == {'tag': 'x', 'weight': 2}

    def test_collection_resolves_no_dataset_params_of_its_children(self):
        class Tagged(OtherDataset):
            fetch_params = (Dimension('tag', 'Tag.', None, default='none'),)

            def _fetch(self, flag, effective_config, tag, **params):
                return hourly_frame(periods=2)

        child = Tagged(name='t')
        collection = DatasetConcatCollection([child])
        collection.fetch(LOAD, config={'param_defaults': {'tag': 'x'}})
        assert collection.get_fetch_record(LOAD, config={'param_defaults': {'tag': 'x'}}).params == {}
        [child_record] = collection.get_fetch_record(LOAD, config={'param_defaults': {'tag': 'x'}}).dependencies
        assert (child_record.params, child_record.param_default_reads) == ({'tag': 'x'}, {'tag': 'x'})

    def test_unknown_keyword_argument_raises(self):
        collection = DatasetConcatCollection([OtherDataset(name='p')])
        with pytest.raises(ValueError, match="no flag spec or dataset class declares the parameters \\['tag'\\]"):
            collection.fetch(LOAD, tag='x')

    def test_collection_passes_explicit_params_only(self):
        collection = DatasetConcatCollection([ParamDataset('a')])
        collection.fetch(PRICE, config={'param_defaults': {'scenario': 'low'}}, solution='first')
        assert collection.datasets[0].seen_params[-1] == {'solution': 'first', 'scenario': 'low', 'threshold': 1}


def test_record_holds_resolved_parameters():
    ds = ParamDataset()
    ds.fetch(PRICE, scenario='low')
    record = ds.get_fetch_record(PRICE, scenario='low')
    assert record.params == {'solution': 'final', 'scenario': 'low', 'threshold': 1}
    assert record.call_kwargs == {'scenario': 'low'}
    assert ds.get_fetch_record(PRICE, scenario='high') is None
    assert isinstance(ds.fetch(PRICE, scenario='low'), pd.DataFrame)


class TestValuesFromTheDataset:
    def test_parameter_values_from_the_dataset_are_checked_inside_the_frame(self):
        fuel = Dimension('fuel', 'Fuel.', lambda ds: {f: f for f in ds.fetch(LOAD).columns}, default='a')

        class FuelDataset(Dataset):
            flag_specs = (series_spec(PRICE, params=(fuel,)), series_spec(LOAD))

            def _fetch(self, flag, effective_config, **params):
                return hourly_frame(periods=2)

        ds = FuelDataset()
        ds.fetch(PRICE, fuel='b')
        assert [d.flag for d in ds.get_fetch_record(PRICE, fuel='b').dependencies] == [LOAD]
        with pytest.raises(ValueError, match="'z' is not a value of 'fuel'"):
            ds.fetch(PRICE, fuel='z')


class TestHelperMethods:
    """`fetch_multiple_flags_and_concat` passes each fetch the parameters its spec declares."""

    def test_multiple_flags_and_concat_passes_parameters_only_to_flags_that_declare_them(self):
        ds = ParamDataset()
        ds.fetch_multiple_flags_and_concat([PRICE, LOAD], solution='first', scenario='low')
        assert ds.seen_params == [{'solution': 'first', 'scenario': 'low', 'threshold': 1}, {}]


class Tagged(OtherDataset):
    """Stores the `tag` that reaches `_fetch`."""

    fetch_params = (Dimension('tag', 'Tag.', None, default='child default'),)

    def __init__(self, name: str = None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.seen_tags: list = []

    def _fetch(self, flag, effective_config, tag, **params):
        self.seen_tags.append(tag)
        return hourly_frame(periods=2)


class TaggedConcat(DatasetConcatCollection):
    """A collection that declares the dataset parameter `tag` of its children itself."""

    fetch_params = (Dimension('tag', 'Tag.', None, default='collection default'),)


class TestCollectionForwardsOnlyGivenValues:
    def test_default_of_the_collection_does_not_reach_the_child(self):
        child = Tagged('t', config=DatasetConfig(param_defaults={'tag': 'child config'}))
        TaggedConcat([child]).fetch(LOAD)
        assert child.seen_tags == ['child config']

    def test_call_and_param_defaults_of_the_collection_reach_the_child(self):
        child = Tagged('t')
        TaggedConcat([child], config=DatasetConfig(param_defaults={'tag': 'collection config'})).fetch(LOAD)
        TaggedConcat([child]).fetch(LOAD, tag='call')
        assert child.seen_tags == ['collection config', 'call']


class TestNestedCollectionsForwardOwnParams:
    """A parameter of a collection in the call and the same value in `param_defaults` reach nested children alike."""

    @staticmethod
    def leaf(name: str, factor: int = 1) -> StaticDataset:
        return StaticDataset({'x': hourly_frame(periods=2) * factor}, name=name)

    def nested_concat(self) -> DatasetConcatCollection:
        inner_1 = DatasetConcatCollection([self.leaf('a'), self.leaf('b')], name='i1')
        inner_2 = DatasetConcatCollection([self.leaf('c'), self.leaf('d')], name='i2')
        return DatasetConcatCollection([inner_1, inner_2], name='outer')

    def comparison_of_comparisons(self) -> DatasetComparison:
        first = DatasetComparison(self.leaf('v1', 3), self.leaf('r1'))
        second = DatasetComparison(self.leaf('v2', 2), self.leaf('r2'))
        return DatasetComparison(first, second)

    def assert_call_equals_param_defaults(self, build, **params):
        in_call, in_config = build(), build()
        config = {'param_defaults': params}
        pd.testing.assert_frame_equal(in_call.fetch('x', **params), in_config.fetch('x', config=config))
        call_children = in_call.get_fetch_record('x', **params).dependencies
        config_children = in_config.get_fetch_record('x', config=config).dependencies
        assert [r.params for r in call_children] == [r.params for r in config_children]
        assert all(r.call_kwargs == params for r in call_children + config_children)

    def test_comparison_of_comparisons(self):
        self.assert_call_equals_param_defaults(
            self.comparison_of_comparisons, comparison_type=ComparisonTypeEnum.VARIATION
        )
        result = self.comparison_of_comparisons().fetch('x', comparison_type=ComparisonTypeEnum.VARIATION)
        pd.testing.assert_frame_equal(result, hourly_frame(periods=2) * 3)

    def test_kwargs_for_child_outside_of_fetch_raises(self):
        collection = self.nested_concat()
        with pytest.raises(RuntimeError, match='no open fetch of this collection'):
            collection._kwargs_for_child(collection.datasets[0], 'x')


class TestParamDefaultNames:
    """Instance and call configs are checked against this dataset, class configs against all dataset classes."""

    def test_class_config_may_name_a_dataset_param_of_another_class(self):
        Dataset.set_class_config(DatasetConfig(param_defaults={'comparison_type': ComparisonTypeEnum.VARIATION}))
        ParamDataset().fetch(LOAD)

    def test_instance_config_must_name_a_param_of_this_dataset(self):
        ds = ParamDataset(config=DatasetConfig(param_defaults={'comparison_type': ComparisonTypeEnum.VARIATION}))
        with pytest.raises(ValueError, match=r"does not declare: \['comparison_type'\]"):
            ds.fetch(LOAD)

    def test_call_config_must_name_a_param_of_this_dataset(self):
        with pytest.raises(ValueError, match=r"does not declare: \['comparison_type'\]"):
            ParamDataset().fetch(LOAD, config={'param_defaults': {'comparison_type': ComparisonTypeEnum.VARIATION}})

    def test_collection_instance_config_may_name_a_dataset_param_of_a_child(self):
        comparison = DatasetComparison(ParamDataset('v'), ParamDataset('r'))
        config = DatasetConfig(param_defaults={'comparison_type': ComparisonTypeEnum.VARIATION, 'scenario': 'low'})
        DatasetConcatCollection([comparison], config=config).fetch(LOAD)


class TestNameClash:
    @staticmethod
    def tag_spec() -> TimeSeriesFlagSpec:
        return series_spec(PRICE, params=(Dimension('tag', 'Tag.', None, default='x'),))

    def test_spec_with_a_dataset_parameter_name_raises_at_construction(self):
        class TaggedWithSpec(Tagged):
            flag_specs = (self.tag_spec(),)

        with pytest.raises(ValueError, match='A parameter name must have one owner'):
            TaggedWithSpec('t')

    def test_spec_added_after_construction_raises_at_fetch(self):
        ds = Tagged('t')
        ds.flag_specs = (*ds.flag_specs, self.tag_spec())
        ds.flag_index.register_new_flag(
            PRICE,
            ItemTypeEnum.TimeSeries,
            unit=Units.MW,
            model_flag=None,
            schema=self.tag_spec().schema,
            params=self.tag_spec().params,
        )
        with pytest.raises(ValueError, match=r"\['tag'\] are flag parameters of the spec and dataset parameters"):
            ds.fetch(PRICE)


class TestTypedFetchOverrides:
    @staticmethod
    def dataset_classes_with_fetch_override() -> list[type]:
        classes, todo = [], [Dataset]
        while todo:
            cls = todo.pop()
            todo.extend(cls.__subclasses__())
            if cls is not Dataset and 'fetch' in cls.__dict__ and cls.__module__.startswith('mesqual.'):
                classes.append(cls)
        return classes

    def test_overrides_exist(self):
        assert set(self.dataset_classes_with_fetch_override()) >= {
            DatasetComparison,
            DatasetConcatCollectionOfComparisons,
        }

    def test_keyword_only_arguments_equal_the_dataset_params(self):
        for cls in self.dataset_classes_with_fetch_override():
            parameters = inspect.signature(cls.fetch).parameters.values()
            keyword_only = {p.name for p in parameters if p.kind is inspect.Parameter.KEYWORD_ONLY}
            declared = {d.name for d in cls.fetch_params}
            if issubclass(cls, DatasetCollection):
                declared |= {d.name for d in cls.get_child_dataset_type().fetch_params}
            declared -= {d.name for d in Dataset.fetch_params}
            assert keyword_only == declared, cls.__name__
            keyword_only_parameters = [p for p in parameters if p.kind is inspect.Parameter.KEYWORD_ONLY]
            assert all(p.default is UNSET for p in keyword_only_parameters)
            assert all(str(p.annotation).endswith('| Unset') for p in keyword_only_parameters), cls.__name__

    def test_parameter_left_out_resolves_from_param_defaults(self):
        config = DatasetConfig(param_defaults={'comparison_type': ComparisonTypeEnum.VARIATION})
        variation = StaticDataset({'x': hourly_frame(periods=2) * 2}, name='var')
        comparison = DatasetComparison(variation, StaticDataset({'x': hourly_frame(periods=2)}, name='ref'))
        collection = DatasetConcatCollectionOfComparisons([comparison], config=config, concat_axis=0)
        result = collection.fetch('x', fill_value=1)
        pd.testing.assert_frame_equal(result.droplevel(0), hourly_frame(periods=2) * 2)
        record = collection.get_fetch_record('x', fill_value=1)
        assert record.call_kwargs == {'fill_value': 1}
        [child] = record.dependencies
        assert child.param_default_reads == {'comparison_type': ComparisonTypeEnum.VARIATION}

    def test_unset_has_a_clear_repr_and_type(self):
        assert repr(UNSET) == '<from param_defaults or default>'
        assert isinstance(UNSET, Unset) and Unset() is UNSET


class TestComparisonFillValue:
    @staticmethod
    def comparison(fill_value=None, **kwargs) -> DatasetComparison:
        spec = series_spec('x', fill_value=fill_value)
        variation = StaticDataset({'x': hourly_frame(periods=2)}, name='var', specs=(spec,))
        reference = StaticDataset({'x': hourly_frame(periods=2)[['a']]}, name='ref', specs=(spec,))
        return DatasetComparison(variation, reference, **kwargs)

    def test_spec_fill_value_applies_without_a_fill_value_parameter(self):
        assert list(self.comparison(fill_value=0).fetch('x')['b']) == [0, 1]
        assert self.comparison(fill_value=None).fetch('x')['b'].isna().all()

    def test_fill_value_parameter_wins_over_the_spec(self):
        assert list(self.comparison(fill_value=0).fetch('x', fill_value=10)['b']) == [-10, -9]

    def test_nan_fill_value_means_no_fill(self):
        assert self.comparison(fill_value=0).fetch('x', fill_value=float('nan'))['b'].isna().all()

    def test_comparison_gives_the_same_numbers_inside_a_concat_of_comparisons(self):
        comparison = self.comparison(fill_value=0, config=DatasetConfig(param_defaults={'fill_value': 5}))
        alone = comparison.fetch('x')
        inside = DatasetConcatCollectionOfComparisons([comparison]).fetch('x')[comparison.name]
        pd.testing.assert_frame_equal(inside, alone)
        assert list(alone['b']) == [-5, -4]
