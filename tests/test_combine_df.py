import pytest
import pandas as pd

from mesqual.utils.pandas_utils.combine_df import combine_dfs


def test_combine_dfs_accepts_a_generator():
    frames = [
        pd.DataFrame({'A': [1, 2]}, index=['x', 'y']),
        pd.DataFrame({'B': [3, 4]}, index=['x', 'y']),
    ]

    result = combine_dfs(df for df in frames)

    expected = pd.DataFrame({'A': [1, 2], 'B': [3, 4]}, index=['x', 'y'])
    pd.testing.assert_frame_equal(result, expected)


def test_combine_dfs_accepts_a_generator_with_one_item():
    frame = pd.DataFrame({'A': [1, 2]}, index=['x', 'y'])
    assert combine_dfs(df for df in [frame]) is frame


class TestPrecedence:
    @staticmethod
    def frames():
        return [
            pd.DataFrame({'A': [1.0, None]}, index=['x', 'y']),
            pd.DataFrame({'A': [10.0, 20.0]}, index=['x', 'y']),
        ]

    def test_first_keeps_the_values_of_the_earlier_input(self):
        result = combine_dfs(self.frames(), precedence='first')
        assert result['A'].tolist() == [1.0, 20.0]

    def test_last_keeps_the_values_of_the_later_input(self):
        result = combine_dfs(self.frames(), precedence='last')
        assert result['A'].tolist() == [10.0, 20.0]

    def test_other_value_raises(self):
        with pytest.raises(ValueError, match='precedence'):
            combine_dfs(self.frames(), precedence=True)
