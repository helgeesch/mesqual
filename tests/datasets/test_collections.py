"""Collections: names, specs, routing, merge precedence, concat levels, sums, comparisons and schemas."""

import numpy as np
import pandas as pd
import pytest

from mesqual import StudyManager
from mesqual.datasets import (
    AmbiguousOverrideError,
    DatasetComparison,
    DatasetConcatCollection,
    DatasetConcatCollectionOfComparisons,
    DatasetConfig,
    DatasetLinkCollection,
    DatasetMergeCollection,
    DatasetSumCollection,
)
from mesqual.enums import ComparisonTypeEnum, TopologyTypeEnum
from mesqual.flag import FrameSchema, Level, ModelFlagSpec, TimeSeriesFlagSpec
from mesqual.units import Units
from tests.datasets.fixtures import StaticDataset, hourly_frame, series_spec

NODES = 'Node.Model'
PRICE = 'Node.Results.price'
NODE_SPEC = ModelFlagSpec(name=NODES, description='Nodes.', index='Node', topology_type=TopologyTypeEnum.Node)
PRICE_SPEC = TimeSeriesFlagSpec(name=PRICE, description='Price.', model_flag=NODES, unit=Units.EUR_per_MWh)
SUMMED_SPEC = TimeSeriesFlagSpec(
    name=PRICE,
    description='Price.',
    model_flag=NODES,
    unit=Units.EUR_per_MWh,
    fill_value=0,
)


def node_dataset(name: str, prices: pd.DataFrame = None, nodes: pd.DataFrame = None, spec=PRICE_SPEC):
    prices = hourly_frame(periods=3) if prices is None else prices
    nodes = pd.DataFrame({'zone': ['DE', 'FR']}, index=pd.Index(['a', 'b'], name='Node')) if nodes is None else nodes
    prices = prices.rename_axis(index='time', columns='Node')
    return StaticDataset({NODES: nodes, PRICE: prices}, name=name, specs=(NODE_SPEC, spec))


def x_dataset(name: str, frame: pd.DataFrame | pd.Series = None, **spec_fields) -> StaticDataset:
    frame = hourly_frame(periods=3) if frame is None else frame
    return StaticDataset({'x': frame}, name=name, specs=(series_spec('x', **spec_fields),))


class TestConstruction:
    @pytest.mark.parametrize(
        'collection_type',
        [
            DatasetConcatCollection,
            DatasetMergeCollection,
            DatasetSumCollection,
            DatasetLinkCollection,
        ],
    )
    def test_duplicate_names_raise(self, collection_type):
        """Two children with the same name raise `ValueError`."""
        with pytest.raises(ValueError, match=r"names \['a'\] are not unique"):
            collection_type([x_dataset('a'), x_dataset('a')], name='c')

    def test_comparison_of_a_dataset_with_itself_raises(self):
        dataset = x_dataset('a')
        with pytest.raises(ValueError, match='not unique'):
            DatasetComparison(dataset, dataset)

    def test_children_with_different_specs_for_one_flag_raise(self):
        """Two children with different specs for one flag raise `ValueError`."""
        with pytest.raises(ValueError, match="Two different specs for 'x'"):
            DatasetConcatCollection([x_dataset('a'), x_dataset('b', unit=Units.MWh)])

    def test_add_dataset_with_a_different_spec_for_one_flag_raises(self):
        first = x_dataset('a')
        collection = DatasetConcatCollection([first])
        with pytest.raises(ValueError, match="Two different specs for 'x'"):
            collection.add_dataset(x_dataset('b', unit=Units.MWh))
        assert collection.datasets == [first]
        assert collection.flag_index.get_spec('x') is first.flag_index.get_spec('x')

    def test_add_dataset_checks_the_specs_without_the_replaced_child(self):
        other = StaticDataset({'y': hourly_frame(periods=2)}, name='b')
        collection = DatasetConcatCollection([x_dataset('a'), other])
        replacement = x_dataset('a', unit=Units.MWh)
        collection.add_dataset(replacement)
        assert collection.datasets == [replacement, other]
        assert collection.flag_index.get_spec('x').unit == Units.MWh

    def test_link_collection_with_two_children_with_one_flag_raises(self):
        """A link collection raises when two children accept the same flag."""
        with pytest.raises(AmbiguousOverrideError, match=r"several children accept the flags \['x'\]"):
            DatasetLinkCollection([x_dataset('a'), x_dataset('b')])

    def test_link_collection_routes_each_flag_to_its_child(self):
        link = DatasetLinkCollection([x_dataset('a'), StaticDataset({'y': hourly_frame(periods=2) * 2}, name='b')])
        pd.testing.assert_frame_equal(link.fetch('x'), hourly_frame(periods=3))
        pd.testing.assert_frame_equal(link.fetch('y'), hourly_frame(periods=2) * 2)


class TestMergePrecedence:
    @staticmethod
    def overlapping() -> list[StaticDataset]:
        return [x_dataset('a', hourly_frame(periods=3)), x_dataset('b', hourly_frame(periods=3) + 100)]

    def test_first_keeps_the_values_of_the_first_child(self):
        merged = DatasetMergeCollection(self.overlapping()).fetch('x')
        pd.testing.assert_frame_equal(merged, hourly_frame(periods=3), check_dtype=False)

    def test_last_keeps_the_values_of_the_last_child(self):
        merged = DatasetMergeCollection(self.overlapping(), precedence='last').fetch('x')
        pd.testing.assert_frame_equal(merged, hourly_frame(periods=3) + 100, check_dtype=False)

    def test_name_list_gives_the_order(self):
        children = self.overlapping() + [x_dataset('c', hourly_frame(periods=3) + 200)]
        merged = DatasetMergeCollection(children, precedence=['b', 'c', 'a']).fetch('x')
        pd.testing.assert_frame_equal(merged, hourly_frame(periods=3) + 100, check_dtype=False)

    def test_partial_overlap_fills_from_the_other_child(self):
        first = hourly_frame(periods=3).astype(float)
        first.iloc[0, 0] = np.nan
        merged = DatasetMergeCollection([x_dataset('a', first), x_dataset('b', hourly_frame(periods=3) + 100)]).fetch(
            'x'
        )
        assert merged.iloc[0, 0] == 100
        assert merged.iloc[1, 0] == 1

    @pytest.mark.parametrize('precedence', [['a'], ['a', 'a', 'b'], ['a', 'x'], 'middle', True])
    def test_invalid_precedence_raises(self, precedence):
        with pytest.raises(ValueError, match='precedence must be'):
            DatasetMergeCollection(self.overlapping(), precedence=precedence)

    def test_name_generator_gives_the_order(self):
        merged = DatasetMergeCollection(self.overlapping(), precedence=(name for name in ['b', 'a']))
        assert merged.precedence == ['b', 'a']
        pd.testing.assert_frame_equal(merged.fetch('x'), hourly_frame(periods=3) + 100, check_dtype=False)

    def test_add_dataset_with_a_name_list_checks_the_precedence(self):
        merged = DatasetMergeCollection(self.overlapping(), precedence=['b', 'a'])
        with pytest.raises(ValueError, match='precedence must be'):
            merged.add_dataset(x_dataset('c'))
        assert [ds.name for ds in merged.datasets] == ['a', 'b']
        merged.precedence = ['c', 'b', 'a']
        merged.add_dataset(x_dataset('c', hourly_frame(periods=3) + 200))
        pd.testing.assert_frame_equal(merged.fetch('x'), hourly_frame(periods=3) + 200, check_dtype=False)

    def test_fetch_merged_takes_precedence(self):
        collection = DatasetConcatCollection(self.overlapping())
        merged = collection.fetch_merged('x', precedence=['b', 'a'])
        pd.testing.assert_frame_equal(merged, hourly_frame(periods=3) + 100, check_dtype=False)
        assert collection.get_merged_dataset_collection('last').precedence == 'last'

    def test_split_scenario_gives_one_continuous_series(self):
        full = hourly_frame(periods=6)
        merged = DatasetMergeCollection([x_dataset('jan', full.iloc[:3]), x_dataset('feb', full.iloc[3:])])
        pd.testing.assert_frame_equal(merged.fetch('x'), full)


class TestConcat:
    @staticmethod
    def children() -> list[StaticDataset]:
        return [node_dataset('a'), node_dataset('b', hourly_frame(periods=3) * 2)]

    def test_default_level_at_the_top_of_the_columns(self):
        result = DatasetConcatCollection(self.children()).fetch(PRICE)
        assert result.columns.names == ['dataset', 'Node']

    def test_custom_level_name(self):
        result = DatasetConcatCollection(self.children(), concat_level_name='scenario').fetch(PRICE)
        assert result.columns.names == ['scenario', 'Node']

    def test_concat_top_false_puts_the_level_last(self):
        """`concat_top=False` puts the new level last."""
        result = DatasetConcatCollection(self.children(), concat_top=False).fetch(PRICE)
        assert result.columns.names == ['Node', 'dataset']
        assert result[('a', 'b')].tolist() == [0, 2, 4]

    def test_concat_top_false_with_three_levels(self):
        """With `concat_top=False`, the new level comes after all levels of the child result."""
        inner = DatasetConcatCollection(self.children(), name='inner')
        outer = DatasetConcatCollection([inner], concat_top=False, concat_level_name='outer')
        assert outer.fetch(PRICE).columns.names == ['dataset', 'Node', 'outer']

    def test_rows_keep_the_order_of_the_children(self):
        children = [node_dataset('z'), node_dataset('a')]
        result = DatasetConcatCollection(children, concat_axis=0).fetch(PRICE)
        assert result.index.names == ['dataset', 'time']
        assert list(result.index.get_level_values('dataset').unique()) == ['z', 'a']

    def test_a_concat_on_columns_sorts_its_rows_also_with_a_child_concat_level_on_rows(self):
        inner = DatasetConcatCollection([node_dataset('z'), node_dataset('a')], name='inner', concat_axis=0)
        outer = DatasetConcatCollection([inner], name='outer')
        assert list(inner.fetch(PRICE).index.get_level_values('dataset').unique()) == ['z', 'a']
        result = outer.fetch(PRICE)
        assert result.index.names == ['dataset', 'time']
        assert result.index.is_monotonic_increasing

    def test_differing_horizons_give_nan(self):
        children = [node_dataset('long', hourly_frame(periods=4)), node_dataset('short', hourly_frame(periods=2))]
        result = DatasetConcatCollection(children).fetch(PRICE)
        assert len(result) == 4
        assert result['short'].iloc[2:].isna().all().all()

    def test_different_level_names_raise(self):
        children = [x_dataset('a'), x_dataset('b', hourly_frame(periods=3).rename_axis(index='other'))]
        with pytest.raises(NotImplementedError, match='Axes names do not match'):
            DatasetConcatCollection(children).fetch('x')

    def test_same_level_names_in_another_order_raise(self):
        """Child results with the same level names in another order raise `ValueError`."""

        def frame(names):
            data = hourly_frame(periods=3)
            data.columns = pd.MultiIndex.from_product([data.columns, ['v']], names=names)
            return data

        children = [x_dataset('a', frame(['Node', 'side'])), x_dataset('b', frame(['side', 'Node']))]
        message = (
            r"Child 'b' has the levels \['side', 'Node'\] on axis 1 in another order than child 'a' \['Node', 'side'\]"
        )
        with pytest.raises(ValueError, match=message):
            DatasetConcatCollection(children).fetch('x')

    def test_shorter_child_with_levels_in_another_order_raises(self):
        long = hourly_frame(periods=3)
        long.columns = pd.MultiIndex.from_product([long.columns, ['v'], ['w']], names=['Node', 'side', 'part'])
        short = hourly_frame(periods=3)
        short.columns = pd.MultiIndex.from_product([['v'], short.columns], names=['side', 'Node'])
        children = [x_dataset('a', long), x_dataset('b', short)]
        message = r"Child 'b' has the levels \['side', 'Node'\] on axis 1 in another order than child 'a'"
        with pytest.raises(ValueError, match=message):
            DatasetConcatCollection(children).fetch('x')

    @pytest.mark.parametrize(
        'long_names, other_names',
        [
            ([None, None, 'part'], ['part', None]),
            (['a', None, None], [None, 'a', None]),
            (['a', 'b', 'c'], ['a', 'a']),
            (['a', None, 'c'], [None, None]),
        ],
    )
    def test_duplicate_level_names_raise_not_implemented(self, long_names, other_names):
        """A result with duplicate level names, for example unnamed levels, raises `NotImplementedError`."""

        def frame(names):
            data = hourly_frame(periods=3)
            extra = [['v']] * (len(names) - 1)
            data.columns = pd.MultiIndex.from_product([data.columns, *extra], names=names)
            return data

        children = [x_dataset('a', frame(long_names)), x_dataset('b', frame(other_names))]
        with pytest.raises(NotImplementedError, match='Axes names do not match'):
            DatasetConcatCollection(children).fetch('x')

    def test_missing_level_is_added_with_an_empty_value(self):
        extra = hourly_frame(periods=3).rename_axis(columns='Node')
        extra.columns = pd.MultiIndex.from_product([extra.columns, ['v']], names=['Node', 'side'])
        children = [x_dataset('plain', hourly_frame(periods=3).rename_axis(columns='Node')), x_dataset('extra', extra)]
        result = DatasetConcatCollection(children).fetch('x')
        assert result.columns.names == ['dataset', 'Node', 'side']
        assert result[('plain', 'a', '')].tolist() == [0, 1, 2]
        assert result[('extra', 'a', 'v')].tolist() == [0, 1, 2]


class TestConcatAxis:
    """`concat_axis` is fixed per collection; each nested concat collection uses its own."""

    @staticmethod
    def group(name: str, concat_axis: int = 1) -> DatasetConcatCollection:
        return DatasetConcatCollection([node_dataset('a'), node_dataset('b')], name=name, concat_axis=concat_axis)

    def test_default_concats_columns(self):
        collection = self.group('g')
        result = collection.fetch(PRICE)
        assert collection.concat_axis == 1
        assert (list(result.index.names), list(result.columns.names), result.shape) == (
            ['time'],
            ['dataset', 'Node'],
            (3, 4),
        )

    def test_axis_0_stacks_rows(self):
        result = self.group('g', concat_axis=0).fetch(PRICE)
        assert (list(result.index.names), list(result.columns.names), result.shape) == (
            ['dataset', 'time'],
            ['Node'],
            (6, 2),
        )

    @pytest.mark.parametrize('concat_axis', [np.int64(0), np.int64(1)])
    def test_numpy_integers_are_accepted(self, concat_axis):
        collection = DatasetConcatCollection([node_dataset('a')], concat_axis=concat_axis)
        assert collection.concat_axis == concat_axis
        assert type(collection.concat_axis) is int

    @pytest.mark.parametrize('concat_axis', [2, -1, None, True, np.True_, 0.0, 1.0, 'index', 'columns'])
    def test_other_values_raise(self, concat_axis):
        with pytest.raises(ValueError, match='concat_axis must be 0 \\(rows\\) or 1 \\(columns\\)'):
            DatasetConcatCollection([node_dataset('a')], concat_axis=concat_axis)

    def test_layout_is_read_only(self):
        collection = self.group('g')
        for name in ('concat_axis', 'concat_top', 'concat_level_name'):
            with pytest.raises(AttributeError):
                setattr(collection, name, None)

    def test_column_groups_in_a_row_parent_keep_their_axis(self):
        outer = DatasetConcatCollection([self.group('g1'), self.group('g2')], concat_level_name='group', concat_axis=0)
        result = outer.fetch(PRICE)
        assert (list(result.index.names), list(result.columns.names)) == (['group', 'time'], ['dataset', 'Node'])
        pd.testing.assert_frame_equal(result.loc['g1'], self.group('g1').fetch(PRICE))

    def test_row_groups_in_a_column_parent_keep_their_axis(self):
        outer = DatasetConcatCollection([self.group('g1', 0), self.group('g2', 0)], concat_level_name='group')
        result = outer.fetch(PRICE)
        assert (list(result.index.names), list(result.columns.names)) == (['dataset', 'time'], ['group', 'Node'])
        pd.testing.assert_frame_equal(result['g1'], self.group('g1', 0).fetch(PRICE))


class TestConcatOfFlags:
    def test_concat_level_at_top_false_puts_the_level_last(self):
        """`concat_level_at_top=False` puts the flag level last."""
        dataset = StaticDataset({'x': hourly_frame(periods=2), 'y': hourly_frame(periods=2)})
        result = dataset.fetch_multiple_flags_and_concat(['x', 'y'], concat_level_at_top=False)
        assert result.columns.names == [None, 'variable']
        assert list(result.columns) == [('a', 'x'), ('b', 'x'), ('a', 'y'), ('b', 'y')]


class TestSum:
    def test_time_series_add_with_the_spec_fill_value(self):
        one = node_dataset('one', hourly_frame(periods=2), spec=SUMMED_SPEC)
        other = node_dataset('other', hourly_frame(periods=2)[['a']] * 10, spec=SUMMED_SPEC)
        result = DatasetSumCollection([one, other]).fetch(PRICE)
        assert result['a'].tolist() == [0, 11]
        assert result['b'].tolist() == [0, 1]

    def test_without_spec_fill_value_a_missing_object_gives_nan(self):
        one = node_dataset('one', hourly_frame(periods=2))
        other = node_dataset('other', hourly_frame(periods=2)[['a']] * 10)
        result = DatasetSumCollection([one, other]).fetch(PRICE)
        assert result['b'].isna().all()

    def test_model_flags_merge_and_the_first_wins(self):
        first = node_dataset('one')
        second = node_dataset(
            'two', nodes=pd.DataFrame({'zone': ['NL', 'BE']}, index=pd.Index(['a', 'c'], name='Node'))
        )
        result = DatasetSumCollection([first, second]).fetch(NODES)
        assert result['zone'].to_dict() == {'a': 'DE', 'b': 'FR', 'c': 'BE'}

    def test_non_numeric_time_series_raise_a_clear_error(self):
        """Non-numeric time series raise `TypeError` that names the children."""
        text = hourly_frame(periods=2).astype(str)
        collection = DatasetSumCollection([x_dataset('a', text), x_dataset('b', text)], name='sum')
        with pytest.raises(TypeError, match=r"non-numeric data in \['a', 'b'\]"):
            collection.fetch('x')


class TestComparison:
    @staticmethod
    def comparison(var, ref, **spec_fields) -> DatasetComparison:
        return DatasetComparison(x_dataset('var', var, **spec_fields), x_dataset('ref', ref, **spec_fields))

    @staticmethod
    def model_comparison(var, ref) -> DatasetComparison:
        """A comparison of the model flag `NODES`."""
        return DatasetComparison(
            StaticDataset({NODES: var}, name='var', specs=(NODE_SPEC,)),
            StaticDataset({NODES: ref}, name='ref', specs=(NODE_SPEC,)),
        )

    def test_delta_uses_the_spec_fill_value(self):
        var, ref = hourly_frame(periods=2) * 3, hourly_frame(periods=2)[['a']]
        result = self.comparison(var, ref, fill_value=0).fetch('x')
        assert result['b'].tolist() == [0, 3]
        assert self.comparison(var, ref).fetch('x')['b'].isna().all()

    def test_delta_fill_value_parameter_wins_over_the_spec(self):
        var, ref = hourly_frame(periods=2) * 3, hourly_frame(periods=2)[['a']]
        result = self.comparison(var, ref, fill_value=0).fetch('x', fill_value=float('nan'))
        assert result['b'].isna().all()

    def test_all_boolean_frames(self):
        """All-boolean frames compare as non-numeric data, with a marker on each changed cell."""
        var = pd.DataFrame({'on': [True, False]}, index=['g1', 'g2'])
        ref = pd.DataFrame({'on': [True, True]}, index=['g1', 'g2'])
        result = self.comparison(var, ref).fetch('x')
        assert result['on'].tolist() == [True, 'False (was True)']

    def test_non_numeric_changes(self):
        var = pd.DataFrame({'fuel': ['gas', 'coal', 'wind'], 'new': [1, 2, 3]}, index=['g1', 'g2', 'g4'])
        ref = pd.DataFrame({'fuel': ['gas', 'gas', 'hydro'], 'old': [4, 5, 6]}, index=['g1', 'g2', 'g3'])
        result = self.comparison(var, ref).fetch('x')
        assert result['fuel'].to_dict() == {
            'g1': 'gas',
            'g2': 'coal (was gas)',
            'g4': 'wind (new)',
            'g3': 'DELETED: hydro',
        }
        assert result['new'].fillna('-').to_dict() == {'g1': '1 (new)', 'g2': '2 (new)', 'g4': '3 (new)', 'g3': '-'}
        assert result['old'].fillna('-').to_dict() == {
            'g1': 'DELETED: 4',
            'g2': 'DELETED: 5',
            'g4': '-',
            'g3': 'DELETED: 6',
        }

    def test_non_numeric_rows_of_one_side_keep_nan(self):
        var = pd.DataFrame({'fuel': ['gas', 'wind', None]}, index=['g1', 'g4', 'g5'])
        ref = pd.DataFrame({'fuel': ['gas', 'hydro', None]}, index=['g1', 'g3', 'g6'])
        result = self.comparison(var, ref).fetch('x')
        assert result.loc['g4', 'fuel'] == 'wind (new)'
        assert result.loc['g3', 'fuel'] == 'DELETED: hydro'
        assert result.loc[['g5', 'g6'], 'fuel'].isna().all()

    def test_non_numeric_columns_of_one_side_keep_nan(self):
        var = pd.DataFrame({'fuel': ['gas', 'gas'], 'new': ['a', None]}, index=['g1', 'g2'])
        ref = pd.DataFrame({'fuel': ['gas', 'gas'], 'old': [None, 'b']}, index=['g1', 'g2'])
        result = self.comparison(var, ref).fetch('x')
        assert result.loc['g1', 'new'] == 'a (new)'
        assert result.loc['g2', 'old'] == 'DELETED: b'
        assert pd.isna(result.loc['g2', 'new'])
        assert pd.isna(result.loc['g1', 'old'])

    def test_non_numeric_changes_with_column_levels(self):
        def frame(data, columns):
            return pd.DataFrame(data, index=['g1', 'g2'], columns=pd.MultiIndex.from_tuples(columns))

        var = frame([['gas', 'x'], ['coal', 'y']], [('fuel', 'main'), ('new', 'main')])
        ref = frame([['gas', 'z'], ['gas', 'w']], [('fuel', 'main'), ('old', 'main')])
        result = self.comparison(var, ref).fetch('x')
        assert result[('fuel', 'main')].tolist() == ['gas', 'coal (was gas)']
        assert result[('new', 'main')].tolist() == ['x (new)', 'y (new)']
        assert result[('old', 'main')].tolist() == ['DELETED: z', 'DELETED: w']

    def test_replace_unchanged_values_in_non_numeric_delta_of_a_model_flag(self):
        var = pd.DataFrame({'fuel': ['gas', 'coal']}, index=['g1', 'g2'])
        ref = pd.DataFrame({'fuel': ['gas', 'gas']}, index=['g1', 'g2'])
        result = self.model_comparison(var, ref).fetch(NODES, replace_unchanged_values_by_nan=True)
        assert pd.isna(result.loc['g1', 'fuel'])
        assert result.loc['g2', 'fuel'] == 'coal (was gas)'

    def test_variation_with_replace_unchanged_values_of_a_model_flag(self):
        var = pd.DataFrame({'a': [1.0, 2.0], 'b': [3.0, 4.0]}, index=['n1', 'n2'])
        ref = var.copy()
        var.iloc[1, 0] = 9.0
        result = self.model_comparison(var, ref).fetch(
            NODES, comparison_type=ComparisonTypeEnum.VARIATION, replace_unchanged_values_by_nan=True
        )
        assert result['a'].isna().tolist() == [True, False]
        assert result['b'].isna().all()

    def test_both_puts_variation_and_reference_side_by_side(self):
        var, ref = hourly_frame(periods=2) * 2, hourly_frame(periods=2)
        result = self.comparison(var, ref).fetch('x', comparison_type=ComparisonTypeEnum.BOTH)
        assert result.columns.names == [None, 'compared']
        assert list(result.columns) == [('a', 'var'), ('a', 'ref'), ('b', 'var'), ('b', 'ref')]
        assert result[('a', 'var')].tolist() == [0, 2]
        pd.testing.assert_index_equal(result.index, var.index)

    @pytest.mark.parametrize(
        'options',
        [
            {'comparison_type': ComparisonTypeEnum.DELTA},
            {'comparison_type': ComparisonTypeEnum.DELTA, 'replace_unchanged_values_by_nan': True},
            {'comparison_type': ComparisonTypeEnum.VARIATION, 'replace_unchanged_values_by_nan': True},
            {'comparison_type': ComparisonTypeEnum.BOTH, 'replace_unchanged_values_by_nan': True},
        ],
    )
    def test_series_work_for_all_types(self, options):
        """A comparison of a model flag given as two `Series` works for every comparison type, also with `replace_unchanged_values_by_nan`."""
        var = pd.Series([1.0, 2.0], index=['n1', 'n2'], name='v')
        ref = pd.Series([1.0, 3.0], index=['n1', 'n2'], name='v')
        result = self.model_comparison(var, ref).fetch(NODES, **options)
        if options['comparison_type'] is ComparisonTypeEnum.BOTH:
            assert list(result.columns) == ['var', 'ref']
            assert result.loc['n2'].tolist() == [2.0, 3.0]
            assert result.loc['n1'].isna().all()
        else:
            assert isinstance(result, pd.Series)
            assert result.name == 'v'
            assert result.loc['n2'] == (-1.0 if options['comparison_type'] is ComparisonTypeEnum.DELTA else 2.0)

    def test_non_numeric_series_delta(self):
        """A non-numeric `DELTA` of two `Series` marks the changed values."""
        var = pd.Series(['gas', 'coal'], index=['g1', 'g2'])
        ref = pd.Series(['gas', 'gas'], index=['g1', 'g2'])
        result = self.comparison(var, ref).fetch('x')
        assert result.tolist() == ['gas', 'coal (was gas)']


class TestComparisonOfTimeSeries:
    @staticmethod
    def comparison() -> DatasetComparison:
        index = pd.date_range('2024-01-01', periods=4, freq='15min', name='datetime')
        ref = pd.DataFrame({'a': [1.0, 1.0, 1.0, 1.0]}, index=index)
        var = pd.DataFrame({'a': [1.0, 1.0, 1.0, 5.0]}, index=index)
        return DatasetComparison(x_dataset('var', var), x_dataset('ref', ref))

    def test_replace_unchanged_values_of_a_time_series_flag_raises(self):
        with pytest.raises(ValueError, match=r'only for flags that are not time series.*delta\.mask\(delta == 0\)'):
            self.comparison().fetch('x', replace_unchanged_values_by_nan=True)

    def test_replace_unchanged_values_of_a_time_series_flag_inside_a_collection_raises(self):
        collection = DatasetConcatCollectionOfComparisons(
            [self.comparison()],
            config=DatasetConfig(param_defaults={'replace_unchanged_values_by_nan': True}),
        )
        with pytest.raises(ValueError, match='only for flags that are not time series'):
            collection.fetch('x')

    def test_granularity_averages_all_deltas(self):
        assert self.comparison().fetch('x', granularity='1h')['a'].tolist() == [1.0]


class TestStudyWithBoth:
    @staticmethod
    def study() -> StudyManager:
        return StudyManager.factory_from_scenarios(
            [node_dataset('base'), node_dataset('high', hourly_frame(periods=3) * 2)],
            [('high', 'base')],
        )

    def test_scen_comp_concats_scenarios_and_comparisons_with_both(self):
        result = self.study().scen_comp.fetch(PRICE, comparison_type=ComparisonTypeEnum.BOTH)
        assert result.columns.names == ['type', 'dataset', 'Node', 'compared']
        assert result[('scenario', 'high', 'a', '')].tolist() == [0, 2, 4]
        assert result[('comparison', 'high vs base', 'a', 'high')].tolist() == [0, 2, 4]
        assert result[('comparison', 'high vs base', 'a', 'base')].tolist() == [0, 1, 2]

    def test_scen_comp_with_delta_has_no_compared_level(self):
        result = self.study().scen_comp.fetch(PRICE)
        assert result.columns.names == ['type', 'dataset', 'Node']
        assert result[('comparison', 'high vs base', 'b')].tolist() == [0, 1, 2]


class TestTransformSchema:
    @staticmethod
    def level_names(result: pd.DataFrame) -> tuple[list, list]:
        return list(result.index.names), list(result.columns.names)

    @staticmethod
    def schema_names(schema: FrameSchema) -> tuple[list, list]:
        return [level.name for level in schema.index], [level.name for level in schema.columns]

    def assert_schema_matches(self, dataset, flag=PRICE, **params):
        schema = dataset.flag_info(flag).schema
        assert self.schema_names(schema) == self.level_names(dataset.fetch(flag, **params))
        return schema

    @staticmethod
    def children() -> list[StaticDataset]:
        return [node_dataset('a'), node_dataset('b')]

    @pytest.mark.parametrize('outer_axis, inner_axis', [(0, 1), (1, 0), (0, 0), (1, 1)])
    def test_nested_concat_with_different_axes(self, outer_axis, inner_axis):
        inner = DatasetConcatCollection(self.children(), name='inner', concat_axis=inner_axis)
        outer = DatasetConcatCollection([inner], name='outer', concat_level_name='group', concat_axis=outer_axis)
        self.assert_schema_matches(outer)

    def test_link_over_a_row_concat_has_the_row_schema(self):
        self.assert_schema_matches(DatasetLinkCollection([DatasetConcatCollection(self.children(), concat_axis=0)]))

    @pytest.mark.parametrize('collection_type', [DatasetMergeCollection, DatasetSumCollection])
    def test_merge_and_sum_keep_the_schema(self, collection_type):
        schema = self.assert_schema_matches(collection_type(self.children()))
        assert schema == node_dataset('a').flag_info(PRICE).schema

    def test_link_keeps_the_schema(self):
        self.assert_schema_matches(DatasetLinkCollection([node_dataset('a')]))

    def test_concat_adds_a_dataset_level(self):
        schema = self.assert_schema_matches(DatasetConcatCollection(self.children(), concat_level_name='scenario'))
        assert schema.columns[0] == Level('scenario', 'dataset')

    def test_concat_of_a_model_table_on_columns_keeps_the_unnamed_column_level(self):
        collection = DatasetConcatCollection(self.children())
        schema = collection.flag_info(NODES).schema
        result = collection.fetch(NODES)
        assert schema.columns == (Level('dataset', 'dataset'), Level(None, 'dimension'))
        assert result.columns.nlevels == len(schema.columns)
        assert result.columns.get_level_values(1).tolist() == ['zone', 'zone']

    def test_concat_on_rows_and_at_the_bottom(self):
        self.assert_schema_matches(DatasetConcatCollection(self.children(), concat_axis=0))
        self.assert_schema_matches(DatasetConcatCollection(self.children(), concat_top=False))

    def test_comparison_with_both_in_param_defaults_adds_the_compared_level(self):
        a, b = self.children()
        comparison = DatasetComparison(
            a, b, config=DatasetConfig(param_defaults={'comparison_type': ComparisonTypeEnum.BOTH})
        )
        schema = self.assert_schema_matches(comparison)
        assert schema.columns[-1] == Level(DatasetComparison.BOTH_LEVEL_NAME, 'dataset')

    def test_comparison_and_concat_of_comparisons(self):
        a, b = self.children()
        comparison = DatasetComparison(a, b)
        self.assert_schema_matches(comparison)
        self.assert_schema_matches(DatasetConcatCollectionOfComparisons([comparison]))

    def test_study_scen_comp_has_type_and_dataset_levels(self):
        study = TestStudyWithBoth.study()
        schema = self.assert_schema_matches(study.scen_comp)
        assert schema.columns[:2] == (Level('type', 'dataset'), Level('dataset', 'dataset'))
