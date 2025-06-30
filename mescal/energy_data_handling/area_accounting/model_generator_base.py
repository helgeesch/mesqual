from typing import Union, Literal, List

from shapely import Polygon, MultiPolygon, Point

from polylabel import polylabel


class GeoModelGeneratorBase:
    REPRESENTATIVE_POINT_METHOD: Literal['polylabel', 'representative_point'] = 'polylabel'
    _polylabel_cache = {}

    def get_representative_area_point(self, geom: Union[Polygon, MultiPolygon]) -> Point:
        if self.REPRESENTATIVE_POINT_METHOD == 'polylabel':
            return self._get_polylabel_point(geom)
        elif self.REPRESENTATIVE_POINT_METHOD == 'representative_point':
            return geom.representative_point()
        else:
            raise ValueError(f'REPRESENTATIVE_POINT_METHOD {self.REPRESENTATIVE_POINT_METHOD} not supported')

    def _get_polylabel_point(self, geom: Union[Polygon, MultiPolygon]) -> Point:
        key = geom.wkt
        if key in self._polylabel_cache:
            return self._polylabel_cache[key]

        if isinstance(geom, MultiPolygon):
            geom = max(geom.geoms, key=lambda g: g.area)

        exterior = list(geom.exterior.coords)
        holes = [list(ring.coords) for ring in geom.interiors]
        rings = [exterior] + holes

        point = Point(polylabel(rings, 1.0))
        self._polylabel_cache[key] = point
        return point

    @staticmethod
    def _compute_representative_point_from_cloud_of_2d_points(points: List[Point]) -> Point:
        """
        Computes the geometric centroid of a cloud of 2D shapely Points.

        - If the input has 1 point, returns that point.
        - If 2 points, returns their midpoint.
        - If ≥3 points, computes the convex hull and returns the polygon centroid.
        """
        from scipy.spatial import ConvexHull
        n = len(points)
        if n == 0:
            raise ValueError("Empty point list")
        if n == 1:
            return points[0]
        if n == 2:
            return Point((points[0].x + points[1].x) / 2, (points[0].y + points[1].y) / 2)

        coords = [(p.x, p.y) for p in points]
        hull = ConvexHull(coords)
        hull_coords = [coords[i] for i in hull.vertices]
        polygon = Polygon(hull_coords)
        return polygon.representative_point()