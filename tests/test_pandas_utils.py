import numpy as np
import pandas as pd
import pytest

from mesqual.utils.pandas_utils import (
    filter_by_model_query,
    flatten_df,
    prepend_model_prop_levels,
    sort_multiindex,
    xs_df,
)

MODEL = pd.DataFrame(
    {'zone': ['DE', 'DE', 'FR'], 'tech': ['gas', 'wind', 'gas']},
    index=pd.Index(['g1', 'g2', 'g3'], name='gen'),
)
TIME = pd.date_range('2024-01-01', periods=2, freq='h', name='time')
GENERATION = pd.DataFrame(np.arange(6.0).reshape(2, 3), index=TIME, columns=MODEL.index)
STUDY = pd.concat({'a': GENERATION, 'b': GENERATION * 10}, axis=1, names=['dataset'])
"""Columns `dataset` and `gen`."""


class TestXsDf:
    @pytest.mark.parametrize('axis', [0, 1])
    def test_a_scalar_key_drops_the_level_and_a_list_keeps_it(self, axis):
        data = STUDY if axis == 1 else STUDY.T
        scalar = xs_df(data, 'b', axis=axis, level='dataset')
        listed = xs_df(data, ['b'], axis=axis, level='dataset')
        expected = GENERATION * 10 if axis == 1 else (GENERATION * 10).T
        pd.testing.assert_frame_equal(scalar, expected)
        pd.testing.assert_frame_equal(listed, pd.concat({'b': expected}, axis=axis, names=['dataset']))


class TestFlattenDf:
    def test_multiindex_columns_melt_to_one_row_per_value(self):
        flat = flatten_df(STUDY)
        assert list(flat.columns) == ['time', 'dataset', 'gen', 'value']
        assert len(flat) == STUDY.size
        b_g2 = flat[(flat['dataset'] == 'b') & (flat['gen'] == 'g2')]
        assert b_g2['time'].tolist() == TIME.tolist()
        assert b_g2['value'].tolist() == [10.0, 40.0]

    def test_unnamed_levels_get_names(self):
        assert list(flatten_df(pd.DataFrame({'x': [1, 2]})).columns) == ['index', 'columns', 'value']
        data = pd.DataFrame(
            [[1]],
            index=pd.MultiIndex.from_tuples([('a', 1)]),
            columns=pd.MultiIndex.from_tuples([('c', 'd')]),
        )
        assert list(flatten_df(data).columns) == [
            'index_level_0',
            'index_level_1',
            'column_level_0',
            'column_level_1',
            'value',
        ]


class TestPrependModelPropLevels:
    def test_the_last_property_is_the_top_level(self):
        result = prepend_model_prop_levels(GENERATION, MODEL, 'zone', 'tech')
        assert result.columns.names == ['tech', 'zone', 'gen']
        assert result.columns.tolist() == [('gas', 'DE', 'g1'), ('wind', 'DE', 'g2'), ('gas', 'FR', 'g3')]
        assert result.to_numpy().tolist() == GENERATION.to_numpy().tolist()

    def test_prepend_to_top_false_appends_the_level_to_a_series_index(self):
        result = prepend_model_prop_levels(GENERATION.iloc[0], MODEL, 'zone', prepend_to_top=False)
        assert result.index.tolist() == [('g1', 'DE'), ('g2', 'DE'), ('g3', 'FR')]
        assert result.tolist() == [0.0, 1.0, 2.0]

    def test_match_on_level_picks_one_of_two_matching_levels(self):
        pairs = pd.DataFrame(
            [[1.0]],
            columns=pd.MultiIndex.from_tuples([('g1', 'g3')], names=['gen_from', 'gen_to']),
        )
        with pytest.warns(UserWarning, match='Multiple levels match'):
            first = prepend_model_prop_levels(pairs, MODEL, 'zone')
        assert first.columns.tolist() == [('DE', 'g1', 'g3')]
        assert prepend_model_prop_levels(pairs, MODEL, 'zone', match_on_level='gen_to').columns.tolist() == [
            ('FR', 'g1', 'g3')
        ]

    def test_unknown_property_and_no_matching_level_raise(self):
        with pytest.raises(ValueError, match='Property unavailable'):
            prepend_model_prop_levels(GENERATION, MODEL, 'fuel')
        with pytest.raises(ValueError, match='No Index Match'):
            prepend_model_prop_levels(GENERATION.rename(columns={'g1': 'g9'}), MODEL, 'zone')


class TestFilterByModelQuery:
    @pytest.mark.parametrize(
        'data, expected',
        [
            (GENERATION, ['g1', 'g3']),
            (GENERATION.iloc[0], ['g1', 'g3']),
            (STUDY, [('a', 'g1'), ('a', 'g3'), ('b', 'g1'), ('b', 'g3')]),
        ],
        ids=['frame', 'series', 'multiindex'],
    )
    def test_keeps_the_objects_that_match_the_query(self, data, expected):
        result = filter_by_model_query(data, MODEL, 'tech == "gas"')
        if isinstance(result, pd.Series):
            pd.testing.assert_series_equal(result.sort_index(), data.loc[expected])
        else:
            pd.testing.assert_frame_equal(result.sort_index(axis=1), data.loc[:, expected])

    def test_empty_query_returns_the_input(self):
        assert filter_by_model_query(GENERATION, MODEL, '') is GENERATION


class TestSortMultiindex:
    def test_custom_order_first_and_the_other_values_after_in_their_order(self):
        index = pd.MultiIndex.from_product([['x', 'y', 'z'], [1, 2]], names=['k', 'n'])
        data = pd.DataFrame({'v': range(6)}, index=index)
        result = sort_multiindex(data, ['z'], 'k')
        assert result.index.get_level_values('k').tolist() == ['z', 'z', 'x', 'x', 'y', 'y']
        assert result['v'].tolist() == [4, 5, 0, 1, 2, 3]

    def test_columns_and_a_flat_index(self):
        by_dataset = sort_multiindex(STUDY, ['b'], 'dataset', axis=1)
        assert by_dataset.columns.get_level_values(0).tolist() == ['b'] * 3 + ['a'] * 3
        pd.testing.assert_frame_equal(by_dataset['b'], GENERATION * 10)
        by_gen = sort_multiindex(GENERATION.T, ['g3'], 0)
        assert by_gen.index.tolist() == ['g3', 'g1', 'g2']
        assert by_gen.loc['g3'].tolist() == [2.0, 5.0]

    def test_other_axis_raises(self):
        with pytest.raises(ValueError, match='axis'):
            sort_multiindex(GENERATION, ['g1'], 0, axis=2)
