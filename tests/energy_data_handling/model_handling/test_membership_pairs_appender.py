import pandas as pd
import pytest

from mesqual.energy_data_handling.model_handling.membership_pairs_appender import (
    StringMembershipPairsAppender,
    TupleMembershipPairsAppender,
)

LINES = pd.DataFrame(
    {'node_from': ['b', 'a'], 'node_to': ['a', 'c'], 'zone_from': ['Z2', 'Z1'], 'zone_to': ['Z1', 'Z1']},
    index=['l1', 'l2'],
)


def all_combos(appender) -> pd.DataFrame:
    result = appender.append_combo_columns(LINES)
    result = appender.append_sorted_combo_columns(result)
    return appender.append_opposite_combo_columns(result)


@pytest.mark.parametrize(
    'appender, pair',
    [
        (StringMembershipPairsAppender(), lambda a, b: f'{a} - {b}'),
        (StringMembershipPairsAppender(separator='|'), lambda a, b: f'{a}|{b}'),
        (TupleMembershipPairsAppender(), lambda a, b: (a, b)),
    ],
)
def test_combo_sorted_and_opposite_columns_for_every_from_to_pair(appender, pair):
    result = all_combos(appender)
    assert list(result.index) == ['l1', 'l2']
    assert list(result.columns[:4]) == list(LINES.columns)
    assert result['node_combo'].tolist() == [pair('b', 'a'), pair('a', 'c')]
    assert result['node_combo_sorted'].tolist() == [pair('a', 'b'), pair('a', 'c')]
    assert result['node_combo_opposite'].tolist() == [pair('a', 'b'), pair('c', 'a')]
    assert result['zone_combo'].tolist() == [pair('Z2', 'Z1'), pair('Z1', 'Z1')]
    assert result['zone_combo_sorted'].tolist() == [pair('Z1', 'Z2'), pair('Z1', 'Z1')]
    assert result['zone_combo_opposite'].tolist() == [pair('Z1', 'Z2'), pair('Z1', 'Z1')]


def test_prefixes_name_the_new_columns():
    appender = StringMembershipPairsAppender(combo_col_suffix=None, combo_col_prefix='pair_of_')
    assert 'pair_of_node' in appender.append_combo_columns(LINES)


@pytest.mark.parametrize('cls', [StringMembershipPairsAppender, TupleMembershipPairsAppender])
@pytest.mark.parametrize('suffix', ['combo_col_suffix', 'sorted_combo_col_suffix', 'opposite_combo_col_suffix'])
def test_a_combo_column_without_suffix_and_prefix_raises(cls, suffix):
    with pytest.raises(ValueError, match='suffix or prefix'):
        cls(**{suffix: None})
