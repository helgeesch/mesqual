from typing import Union, Literal, List

from shapely import MultiPoint, MultiPolygon, Point, Polygon
from polylabel import polylabel


class GeoModelGeneratorBase:
    """Base class for generating geometric models with representative points.

    This class provides common functionality for working with geometric representations
    of energy system areas, including methods for computing representative points
    within polygons and multipolygons. It's designed to support energy market
    analysis where spatial aggregation of nodes into areas is required.

    The class supports two methods for computing representative points:

    - 'polylabel': The pole of inaccessibility, the point inside the polygon farthest from its boundary
    - 'representative_point': Uses Shapely's built-in representative point method

    Attributes:
        REPRESENTATIVE_POINT_METHOD (str): Method used for computing representative
            points ('polylabel' or 'representative_point')
    """

    REPRESENTATIVE_POINT_METHOD: Literal['polylabel', 'representative_point'] = 'polylabel'

    def get_representative_area_point(self, geom: Union[Polygon, MultiPolygon]) -> Point:
        """Get a representative point for a polygon or multipolygon geometry.

        This method computes a point that is guaranteed to be inside the geometry
        and is suitable for label placement or other visualization purposes in
        energy system maps. For MultiPolygons, it operates on the largest constituent.

        Args:
            geom: A Shapely Polygon or MultiPolygon geometry representing an
                energy system area (e.g., bidding zone, market region)

        Returns:
            Point: A Shapely Point guaranteed to be inside the input geometry,
                suitable for map labels or representative location analysis

        Raises:
            ValueError: If REPRESENTATIVE_POINT_METHOD is not supported

        Examples:
            >>> from shapely.geometry import box
            >>> GeoModelGeneratorBase().get_representative_area_point(box(0, 0, 10, 10))
            <POINT (5 5)>
        """
        if self.REPRESENTATIVE_POINT_METHOD == 'polylabel':
            return self._get_polylabel_point(geom)
        elif self.REPRESENTATIVE_POINT_METHOD == 'representative_point':
            return geom.representative_point()
        else:
            raise ValueError(f'REPRESENTATIVE_POINT_METHOD {self.REPRESENTATIVE_POINT_METHOD} not supported')

    def _get_polylabel_point(self, geom: Union[Polygon, MultiPolygon]) -> Point:
        """Compute representative point using the polylabel algorithm.

        The polylabel algorithm finds the pole of inaccessibility - the most distant
        internal point from the polygon outline. This is particularly useful for
        placing labels on complex energy system area geometries.

        For MultiPolygons, operates on the largest polygon by area, which is
        typically the main landmass for country/region representations.

        Args:
            geom: A Shapely Polygon or MultiPolygon geometry

        Returns:
            Point: The pole of inaccessibility point, cached for performance

        Note:
            Results are cached using the geometry's WKT representation as key.
            The precision parameter (1.0) provides good balance between accuracy
            and performance for typical energy system area sizes.
        """
        key = geom.wkt
        cache: dict[str, Point] = self.__dict__.setdefault('_polylabel_cache', {})
        if key in cache:
            return cache[key]

        if isinstance(geom, MultiPolygon):
            geom = max(geom.geoms, key=lambda g: g.area)

        exterior = list(geom.exterior.coords)
        holes = [list(ring.coords) for ring in geom.interiors]
        rings = [exterior] + holes

        point = Point(polylabel(rings, 1.0))
        cache[key] = point
        return point

    @staticmethod
    def _compute_representative_point_from_cloud_of_2d_points(points: List[Point]) -> Point:
        """Compute a representative point of a collection of 2D points.

        This method is particularly useful in energy systems analysis for computing
        representative locations of energy assets (e.g., power plants, substations)
        that belong to the same area or region.

        The algorithm adapts based on the number of input points:
        - 1 point: Returns the point itself
        - 2 points: Returns the midpoint
        - 3 or more points: Returns the representative point of their convex
          hull, a point inside the hull (a polygon, or a line for collinear points)

        Args:
            points: List of Shapely Point objects representing energy asset
                locations or other spatial features within an area

        Returns:
            Point: Representative point for the collection of input points

        Raises:
            ValueError: If the input list is empty

        Examples:
            >>> from shapely.geometry import Point
            >>> GeoModelGeneratorBase._compute_representative_point_from_cloud_of_2d_points([Point(0, 0), Point(2, 2)])
            <POINT (1 1)>
        """
        n = len(points)
        if n == 0:
            raise ValueError('Empty point list provided - cannot compute representative point')
        if n == 1:
            return points[0]
        if n == 2:
            return Point((points[0].x + points[1].x) / 2, (points[0].y + points[1].y) / 2)

        return MultiPoint(points).convex_hull.representative_point()
