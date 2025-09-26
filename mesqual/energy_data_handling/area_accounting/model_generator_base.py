from typing import Union, Literal, List

from shapely import Polygon, MultiPolygon, Point
from polylabel import polylabel


class GeoModelGeneratorBase:
    """Base class for generating geometric models with representative points.
    
    This class provides common functionality for working with geometric representations
    of energy system areas, including methods for computing representative points
    within polygons and multipolygons. It's designed to support energy market
    analysis where spatial aggregation of nodes into areas is required.
    
    The class supports two methods for computing representative points:
    - 'polylabel': Uses pole of inaccessibility algorithm for optimal label placement
    - 'representative_point': Uses Shapely's built-in representative point method
    
    Attributes:
        REPRESENTATIVE_POINT_METHOD (str): Method used for computing representative
            points ('polylabel' or 'representative_point')
        _polylabel_cache (dict): Cache for expensive polylabel calculations
    """
    REPRESENTATIVE_POINT_METHOD: Literal['polylabel', 'representative_point'] = 'polylabel'
    _polylabel_cache = {}

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
            
        Example:
            
            >>> from shapely.geometry import Polygon, Point
            >>> generator = GeoModelGeneratorBase()
            >>> area_polygon = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
            >>> rep_point = generator.get_representative_area_point(area_polygon)
            >>> print(f"Representative point: {rep_point.x:.2f}, {rep_point.y:.2f}")
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
        """Compute geometric centroid from a collection of 2D points.
        
        This method is particularly useful in energy systems analysis for computing
        representative locations of energy assets (e.g., power plants, substations)
        that belong to the same area or region.
        
        The algorithm adapts based on the number of input points:
        - 1 point: Returns the point itself
        - 2 points: Returns the midpoint
        - ≥3 points: Computes convex hull and returns polygon centroid
        
        Args:
            points: List of Shapely Point objects representing energy asset
                locations or other spatial features within an area
        
        Returns:
            Point: Representative point for the collection of input points
        
        Raises:
            ValueError: If the input list is empty
            
        Example:
            
            >>> from shapely.geometry import Point
            >>> power_plants = [Point(1, 1), Point(3, 2), Point(2, 4)]
            >>> centroid = GeoModelGeneratorBase._compute_representative_point_from_cloud_of_2d_points(power_plants)
            >>> print(f"Regional centroid: {centroid.x:.2f}, {centroid.y:.2f}")
        """
        from scipy.spatial import ConvexHull
        
        n = len(points)
        if n == 0:
            raise ValueError("Empty point list provided - cannot compute representative point")
        if n == 1:
            return points[0]
        if n == 2:
            return Point((points[0].x + points[1].x) / 2, (points[0].y + points[1].y) / 2)

        # For 3+ points, compute convex hull and return centroid
        coords = [(p.x, p.y) for p in points]
        hull = ConvexHull(coords)
        hull_coords = [coords[i] for i in hull.vertices]
        polygon = Polygon(hull_coords)
        return polygon.representative_point()
