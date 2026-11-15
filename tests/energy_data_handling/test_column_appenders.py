import pandas as pd
import pytest

from mesqual.energy_data_handling.variable_utils import AggregatedColumnAppender, UpDownNetAppender

NAN = float('nan')


class TestAggregatedColumnAppender:
    def test_sums_the_columns_with_the_common_part_and_an_all_nan_row_gives_nan(self):
        data = pd.DataFrame({'wind_on': [1.0, NAN, NAN], 'wind_off': [2.0, 5.0, NAN], 'load': [9.0, 9.0, 9.0]})
        result = AggregatedColumnAppender('wind', agg_col_name_suffix='_total').add_aggregated_column(data)
        assert list(result.columns) == ['wind_on', 'wind_off', 'load', 'wind_total']
        pd.testing.assert_series_equal(result['wind_total'], pd.Series([3.0, 5.0, NAN], name='wind_total'))

    def test_multiindex_columns_sum_per_combination_of_the_lower_levels(self):
        columns = pd.MultiIndex.from_product([['st_a', 'st_b', 'gas'], ['DE', 'FR']])
        data = pd.DataFrame([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [NAN, 2.0, NAN, 4.0, 5.0, 6.0]], columns=columns)
        result = AggregatedColumnAppender('st', agg_col_name_prefix='total_').add_aggregated_column(data)
        pd.testing.assert_frame_equal(result['total_st'], pd.DataFrame({'DE': [4.0, NAN], 'FR': [6.0, 6.0]}))


class TestUpDownNetAppender:
    DATA = pd.DataFrame({'X_up': [3.0, 1.0], 'X_down': [1.0, NAN], 'Y': [7.0, 7.0]})

    def test_net_and_total_per_pair_count_a_missing_side_as_zero(self):
        appender = UpDownNetAppender()
        result = appender.append_total_columns_from_up_down_columns(
            appender.append_net_columns_from_up_down_columns(self.DATA)
        )
        assert list(result.columns) == ['X_up', 'X_down', 'Y', 'X_net', 'X_total']
        pd.testing.assert_series_equal(result['X_net'], pd.Series([2.0, 1.0], name='X_net'))
        pd.testing.assert_series_equal(result['X_total'], pd.Series([4.0, 1.0], name='X_total'))

    def test_multiindex_columns_give_net_per_lower_level(self):
        data = pd.concat(
            {'X_up': pd.DataFrame({'DE': [3.0], 'FR': [1.0]}), 'X_down': pd.DataFrame({'DE': [1.0], 'FR': [4.0]})},
            axis=1,
        )
        appender = UpDownNetAppender(net_col_suffix=None, net_col_prefix='net_')
        result = appender.append_net_columns_from_up_down_columns(data)
        pd.testing.assert_frame_equal(result['net_X'], pd.DataFrame({'DE': [2.0], 'FR': [-3.0]}))

    @pytest.mark.parametrize(
        'kwargs',
        [
            {'net_col_suffix': None},
            {'total_col_suffix': ''},
        ],
    )
    def test_a_column_name_without_suffix_and_prefix_raises(self, kwargs):
        with pytest.raises(AttributeError, match='suffix or'):
            UpDownNetAppender(**kwargs)
