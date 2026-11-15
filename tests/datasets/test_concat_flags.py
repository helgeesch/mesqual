import datetime

import pandas as pd
import pytest

from mesqual.datasets import DatasetComparison, DatasetConcatCollection
from mesqual.enums import ComparisonTypeEnum
from tests.datasets.fixtures import StaticDataset, hourly_frame

NET_POSITION = hourly_frame(periods=2)
PER_PARTNER = pd.concat({'p1': NET_POSITION, 'p2': NET_POSITION}, axis=1, names=['partner'])
DAILY = NET_POSITION.set_axis(pd.Index([datetime.date(2024, 1, 1), datetime.date(2024, 1, 2)], name='date'))


def concat(data: dict, **kwargs):
    return StaticDataset(data).fetch_multiple_flags_and_concat(list(data), **kwargs)


class TestSameAxes:
    def test_matching_frames_concat(self):
        result = concat({'a': NET_POSITION, 'b': NET_POSITION * 2})
        assert result.columns.nlevels == 2
        assert list(result.columns.get_level_values('variable').unique()) == ['a', 'b']

    def test_other_number_of_column_levels_raises(self):
        with pytest.raises(ValueError, match=r"'net_position' and 'per_partner'.*columns has 1 and 2 levels"):
            concat({'net_position': NET_POSITION, 'per_partner': PER_PARTNER})

    @pytest.mark.parametrize('flag_concat_axis', [0, 1])
    def test_other_index_type_raises(self, flag_concat_axis):
        with pytest.raises(ValueError, match=r"'hourly' and 'daily'.*DatetimeIndex.*Index"):
            concat({'hourly': NET_POSITION, 'daily': DAILY}, flag_concat_axis=flag_concat_axis)

    def test_series_concat(self):
        result = concat({'a': NET_POSITION['a'], 'b': NET_POSITION['b']})
        assert list(result.columns) == ['a', 'b']

    def test_series_with_other_index_type_raises(self):
        with pytest.raises(ValueError, match='index types'):
            concat({'hourly': NET_POSITION['a'], 'daily': DAILY['a']})

    def test_series_and_frame_raise(self):
        with pytest.raises(ValueError, match='one is a Series, the other a DataFrame'):
            concat({'series': NET_POSITION['a'], 'frame': NET_POSITION})


class TestParameters:
    def test_dataset_param_reaches_each_fetch(self):
        variation = StaticDataset({'a': NET_POSITION * 3, 'b': NET_POSITION * 2}, name='var')
        comparison = DatasetComparison(variation, StaticDataset({'a': NET_POSITION, 'b': NET_POSITION}, name='ref'))
        result = comparison.fetch_multiple_flags_and_concat(['a', 'b'], comparison_type=ComparisonTypeEnum.VARIATION)
        pd.testing.assert_frame_equal(result['a'], NET_POSITION * 3)
        assert result.columns.names[0] == 'variable'

    def test_rows_of_a_row_concat_collection(self):
        data = {'a': NET_POSITION, 'b': NET_POSITION * 2}
        collection = DatasetConcatCollection(
            [StaticDataset(data, name='s1'), StaticDataset(data, name='s2')],
            concat_axis=0,
        )
        result = collection.fetch_multiple_flags_and_concat(['a', 'b'])
        assert list(result.index.names) == ['dataset', 'datetime']
        assert result.columns.names[0] == 'variable'

    def test_undeclared_keyword_argument_raises(self):
        with pytest.raises(ValueError, match=r"no dataset class declares the parameters \['unknown'\]"):
            concat({'a': NET_POSITION}, unknown='x')
