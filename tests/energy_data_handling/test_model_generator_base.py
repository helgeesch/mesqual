from typing import Callable

import geopandas as gpd
import pandas as pd
import pytest
from shapely import Point, box

from mesqual.energy_data_handling.area_accounting import AreaBorderGeometryCalculator, AreaModelGenerator
from mesqual.energy_data_handling.area_accounting.model_generator_base import GeoModelGeneratorBase


class TestRepresentativePoint:
    def test_three_or_more_points_give_a_point_inside_their_hull(self):
        points = [Point(0, 0), Point(4, 0), Point(0, 4), Point(1, 1)]
        point = GeoModelGeneratorBase._compute_representative_point_from_cloud_of_2d_points(points)
        assert point.x + point.y < 4 and point.x > 0 and point.y > 0

    def test_collinear_points(self):
        points = [Point(0, 0), Point(1, 1), Point(2, 2)]
        point = GeoModelGeneratorBase._compute_representative_point_from_cloud_of_2d_points(points)
        assert point.x == point.y


def area_model_generator() -> AreaModelGenerator:
    nodes = pd.DataFrame({'area': ['A', 'B'], 'location': [Point(0, 0), Point(1, 1)]}, index=['n1', 'n2'])
    return AreaModelGenerator(nodes, 'area', 'location')


def border_geometry_calculator() -> AreaBorderGeometryCalculator:
    areas = gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)], index=['A', 'B'], crs='EPSG:4326')
    return AreaBorderGeometryCalculator(areas)


class TestPolylabelCache:
    @pytest.mark.parametrize('factory', [GeoModelGeneratorBase, area_model_generator, border_geometry_calculator])
    def test_instances_do_not_share_the_cache(self, factory: Callable[[], GeoModelGeneratorBase]):
        """Each instance has its own polylabel cache."""
        first, second = factory(), factory()
        first.get_representative_area_point(box(0, 0, 4, 4))
        assert len(first._polylabel_cache) == 1
        assert getattr(second, '_polylabel_cache', {}) == {}

    def test_subclass_without_super_init_gets_a_point(self):
        class OwnInit(GeoModelGeneratorBase):
            def __init__(self):
                self.name = 'own'

        assert OwnInit().get_representative_area_point(box(0, 0, 4, 4)) == Point(2, 2)
