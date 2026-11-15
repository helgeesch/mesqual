"""Border geometry of areas from their polygons."""

import geopandas as gpd
from shapely.geometry import LineString, Point, box

from mesqual.energy_data_handling.area_accounting import AreaBorderGeometryCalculator


class TestAreasWithoutCrs:
    AREAS = gpd.GeoDataFrame(
        geometry=[box(5, 50, 6, 51), box(8, 50, 9, 51), box(6.5, 49, 7.5, 52)], index=['A', 'B', 'C']
    )
    """A and B do not touch. C lies between them, so the border path goes around C. No CRS."""

    def calculator(self) -> AreaBorderGeometryCalculator:
        calculator = AreaBorderGeometryCalculator(self.AREAS.copy())
        calculator.non_crossing_path_finder.show_progress = False
        return calculator

    def test_polygons_without_a_crs_are_lon_lat(self):
        assert self.calculator().area_model_gdf.crs == 'EPSG:4326'

    def test_border_between_areas_that_do_not_touch_gets_a_path(self):
        geometry = self.calculator().calculate_border_geometry('A', 'B')
        assert not geometry['areas_touch']
        assert isinstance(geometry['geo_line_string'], LineString)
        assert geometry['projection_point'] is not None


class TestLineDirection:
    AREAS = gpd.GeoDataFrame(geometry=[box(5, 50, 6, 51), box(8, 50, 9, 51)], index=['A', 'B'], crs='EPSG:4326')
    """A and B do not touch."""

    def test_line_of_each_border_starts_nearer_its_area_from(self):
        calculator = AreaBorderGeometryCalculator(self.AREAS.copy())
        for area_from, area_to in [('A', 'B'), ('B', 'A')]:
            start = Point(calculator.calculate_border_geometry(area_from, area_to)['geo_line_string'].coords[0])
            polygon_from, polygon_to = (self.AREAS.geometry[area] for area in (area_from, area_to))
            assert start.distance(polygon_from) < start.distance(polygon_to), (area_from, area_to)
