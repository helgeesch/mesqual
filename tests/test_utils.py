from importlib.resources import as_file, files

import geopandas as gpd
import pandas as pd

from mesqual.utils.check_all_same import all_same_object
from mesqual.utils.pandas_utils.is_numeric import pd_is_numeric
from mesqual.utils.pandas_utils.move_level import move_top_level_last
from mesqual.utils.set_aggregations import nested_intersection, nested_union


class TestSetHelpers:
    """The helpers turn a generator input into a list once."""

    def test_all_same_object_with_a_generator_of_different_objects(self):
        assert all_same_object(x for x in ([1], [1])) is False

    def test_all_same_object_with_the_same_object(self):
        item = [1]
        assert all_same_object(x for x in (item, item)) is True

    def test_nested_union_with_a_generator(self):
        assert nested_union(s for s in ({1, 2}, {2, 3})) == {1, 2, 3}

    def test_nested_intersection_with_a_generator(self):
        assert nested_intersection(s for s in ({1, 2}, {2, 3})) == {2}

    def test_empty_inputs(self):
        assert nested_union([]) == set()
        assert nested_intersection(iter([])) == set()


class TestPdIsNumeric:
    """The check reads dtypes, and booleans are not numeric."""

    def test_booleans_are_not_numeric(self):
        assert pd_is_numeric(pd.DataFrame({'a': [True, False]})) is False
        assert pd_is_numeric(pd.Series([True, False])) is False

    def test_numbers_are_numeric(self):
        assert pd_is_numeric(pd.DataFrame({'a': [1, 2], 'b': [1.5, float('nan')]})) is True
        assert pd_is_numeric(pd.Series([1.0, 2.0])) is True

    def test_strings_are_not_numeric(self):
        assert pd_is_numeric(pd.DataFrame({'a': [1.0, 2.0], 'b': ['x', 'y']})) is False


class TestMoveTopLevelLast:
    def test_columns_with_three_levels(self):
        columns = pd.MultiIndex.from_tuples([('x', 'a', 1), ('y', 'b', 2)], names=['top', 'mid', 'low'])
        result = move_top_level_last(pd.DataFrame([[1, 2]], columns=columns), axis=1)
        assert list(result.columns.names) == ['mid', 'low', 'top']
        assert list(result.columns) == [('a', 1, 'x'), ('b', 2, 'y')]

    def test_series_rows(self):
        index = pd.MultiIndex.from_tuples([('x', 'a')], names=['top', 'low'])
        assert list(move_top_level_last(pd.Series([1], index=index)).index.names) == ['low', 'top']


class TestPackageData:
    def test_countries_geojson_is_package_data(self):
        resource = files('mesqual.data') / 'countries.geojson'
        assert resource.is_file()
        with as_file(resource) as path:
            gdf = gpd.read_file(path)
        assert not gdf.empty
