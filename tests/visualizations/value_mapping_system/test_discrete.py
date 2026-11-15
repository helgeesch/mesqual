import numpy as np
import pandas as pd
import pytest

from mesqual.visualizations.value_mapping_system.discrete import DiscreteColorMapping, DiscreteLineWidthMapping

PALETTE = ['#111111', '#222222', '#333333']


def test_fixed_and_assigned_outputs_do_not_use_default():
    mapping = DiscreteColorMapping(mapping={'a': '#ff0000'}, palette=PALETTE)
    assert mapping('a') == '#ff0000'
    assert mapping('b') == '#111111'
    assert not mapping.default_used


def test_palette_cycles_when_it_runs_out():
    mapping = DiscreteColorMapping(palette=PALETTE)
    assert [mapping(v) for v in range(5)] == PALETTE + PALETTE[:2]


def test_palette_skips_outputs_of_fixed_entries():
    mapping = DiscreteColorMapping(mapping={'a': '#222222'}, palette=PALETTE)
    assert [mapping(v) for v in ['x', 'y', 'z']] == ['#111111', '#333333', '#111111']


def test_palette_with_only_fixed_outputs_uses_all_outputs():
    mapping = DiscreteColorMapping(mapping={'a': '#111111', 'b': '#222222'}, palette=['#111111', '#222222'])
    assert [mapping(v) for v in ['x', 'y', 'z']] == ['#111111', '#222222', '#111111']


@pytest.mark.parametrize('palette', [None, []])
def test_without_palette_unknown_value_gets_default(palette):
    mapping = DiscreteColorMapping(palette=palette, default_output='#cccccc')
    assert mapping('x') == '#cccccc'
    assert mapping.default_used
    assert mapping.mapping == {}


def test_assignment_is_stable_over_many_frames():
    mapping = DiscreteColorMapping(mapping={'a': '#ff0000'}, palette=PALETTE)
    first = pd.Series(['a', 'b', 'c']).map(mapping)
    second = pd.Series(['c', 'd', 'b']).map(mapping)
    assert list(first) == ['#ff0000', '#111111', '#222222']
    assert list(second) == ['#222222', '#333333', '#111111']
    assert list(mapping.mapping.items()) == [('a', '#ff0000'), ('b', '#111111'), ('c', '#222222'), ('d', '#333333')]


def test_given_mapping_dict_is_not_changed():
    fixed = {'a': '#ff0000'}
    DiscreteColorMapping(mapping=fixed, palette=PALETTE)('b')
    assert fixed == {'a': '#ff0000'}


@pytest.mark.parametrize('value', [None, float('nan'), np.nan, pd.NA, pd.NaT])
def test_missing_values_map_to_default(value):
    mapping = DiscreteColorMapping(palette=PALETTE, default_output='#cccccc')
    assert mapping(value) == '#cccccc'
    assert mapping.default_used
    assert mapping.mapping == {}
    assert mapping('x') == '#111111'


def test_tuple_key_works():
    mapping = DiscreteLineWidthMapping(mapping={('a', 1): 3.0}, palette=[1.0])
    assert mapping(('a', 1)) == 3.0
    assert mapping(('b', 2)) == 1.0


@pytest.mark.parametrize('palette', [tuple(PALETTE), np.array(PALETTE), pd.Series(PALETTE)])
def test_palette_accepts_tuple_array_and_series(palette):
    mapping = DiscreteColorMapping(palette=palette)
    assert [mapping(v) for v in ['x', 'y']] == PALETTE[:2]


@pytest.mark.parametrize('key', [None, float('nan'), pd.NA])
def test_missing_value_as_mapping_key_raises(key):
    with pytest.raises(ValueError, match='default_output'):
        DiscreteColorMapping(mapping={'a': '#ff0000', key: '#000000'})
