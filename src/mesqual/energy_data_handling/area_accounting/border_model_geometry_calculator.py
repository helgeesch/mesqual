from typing import ClassVar, Tuple, Union
import math
import itertools
from tqdm import tqdm
import geopandas as gpd
from pyproj import Transformer
from shapely.geometry import Point, MultiPoint, LineString, MultiLineString, Polygon, MultiPolygon, GeometryCollection
from shapely.ops import nearest_points, linemerge

from mesqual.energy_data_handling.area_accounting.model_generator_base import GeoModelGeneratorBase


class AreaBorderGeometryCalculator(GeoModelGeneratorBase):
    """Geometry of the border between two areas, for maps of border variables.

    Areas that touch have a physical border: the shared boundary line.
    Areas that do not touch, for example across the sea, have a logical
    border: the straight line between the boundaries of the largest polygons
    of the two areas, on the line between their representative points. If
    this line crosses other areas, the line of `NonCrossingPathFinder`
    replaces it when the finder finds one; the finder states its rule. For
    each border, the calculator gives a line, a point for labels and arrows,
    and the azimuth angle of the direction from the first to the second area.

    Attributes:
        area_model_gdf (gpd.GeoDataFrame): GeoDataFrame with area polygon geometries
        non_crossing_path_finder (NonCrossingPathFinder): Path finder for logical borders

    Examples:
        >>> import geopandas as gpd
        >>> countries = gpd.GeoDataFrame(dataset.fetch('Country.Model'), crs='EPSG:4326')
        >>> calculator = AreaBorderGeometryCalculator(countries)
        >>> calculator.calculate_border_geometry('C1', 'C2')
        {'projection_point': <POINT (9 48)>, 'azimuth_angle': 90.0, 'geo_line_string': <LINESTRING (9 50, 9 46)>, 'areas_touch': True}

        Two areas that do not touch get a logical border:

        >>> from shapely.geometry import box
        >>> islands = gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1), box(3, 0, 4, 1)], index=['A', 'B'], crs='EPSG:4326')
        >>> AreaBorderGeometryCalculator(islands).calculate_border_geometry('A', 'B')['geo_line_string']
        <LINESTRING (1 0.5, 3 0.5)>
    """

    PROJECTION_POINT_IDENTIFIER = 'projection_point'
    AZIMUTH_ANGLE_IDENTIFIER = 'azimuth_angle'
    BORDER_AREAS_TOUCH_IDENTIFIER = 'areas_touch'
    BORDER_LINE_STRING_IDENTIFIER = 'geo_line_string'
    GEOMETRY_COLUMNS: ClassVar[tuple[str, ...]] = (
        PROJECTION_POINT_IDENTIFIER,
        AZIMUTH_ANGLE_IDENTIFIER,
        BORDER_AREAS_TOUCH_IDENTIFIER,
        BORDER_LINE_STRING_IDENTIFIER,
    )
    """The keys of `calculate_border_geometry`, in this order."""

    def __init__(self, area_model_gdf: gpd.GeoDataFrame, non_crossing_path_finder: 'NonCrossingPathFinder' = None):
        """Initialize the border geometry calculator.

        Args:
            area_model_gdf: GeoDataFrame containing area geometries with polygon
                boundaries. Index should contain area identifiers (e.g., country codes,
                bidding zone names). Must contain valid polygon geometries in 'geometry' column.
                Without a CRS, the polygons are lon/lat coordinates in EPSG:4326.
                The path of a border between areas that do not touch needs the CRS.
            non_crossing_path_finder: Optional custom path finder for logical borders.
                If None, creates default NonCrossingPathFinder with standard parameters.

        Invalid geometries are cleaned with `buffer(0)`. If `area_model_gdf`
        has a CRS, the cleaning changes it in place.
        """
        self.area_model_gdf = area_model_gdf if area_model_gdf.crs is not None else area_model_gdf.set_crs('EPSG:4326')
        self.non_crossing_path_finder = non_crossing_path_finder or NonCrossingPathFinder()
        self._validate_geometries()

        self._centroid_cache: dict[str, Point] = {}
        self._line_cache: dict[Tuple[str, str], LineString] = {}
        crs = self.area_model_gdf.crs
        self._to_lon_lat = None if crs.is_geographic else Transformer.from_crs(crs, 'EPSG:4326', always_xy=True)

    def _validate_geometries(self):
        """Validate and clean area geometries for reliable calculations.

        Applies buffer(0) operation to fix invalid geometries (self-intersections,
        unclosed rings, etc.) that could cause calculation failures. This is
        particularly important for real-world geographic data that may have
        topology issues.

        Note:
            The buffer(0) operation is a common technique for fixing invalid
            polygon geometries without changing their fundamental shape.
        """
        self.area_model_gdf['geometry'] = self.area_model_gdf['geometry'].apply(
            lambda geom: geom if geom.is_valid else geom.buffer(0)
        )

    def calculate_border_geometry(
        self, area_from: str, area_to: str
    ) -> dict[str, Union[Point, float, LineString, bool]]:
        """Calculate the geometric properties of an area border.

        The steps:

        1. Detect if the areas share a physical boundary (touching or intersecting).
        2. For physical borders: extract the shared boundary line.
        3. For logical borders: compute a connection path.
        4. Calculate the representative point for labels and arrows.
        5. Compute the azimuth angle for arrow icons.

        Args:
            area_from: Source area identifier (must exist in area_model_gdf index)
            area_to: Target area identifier (must exist in area_model_gdf index)

        Returns:
            dict: The border geometry with the keys `projection_point` (Point
                for labels and arrows), `azimuth_angle` (float, direction in
                degrees from 0 to 360), `geo_line_string` (LineString, the border
                line) and `areas_touch` (bool, True when the areas touch or
                overlap, False for logical borders).

        Raises:
            KeyError: If area_from or area_to not found in area_model_gdf
        """
        midpoint, angle = self.get_area_border_midpoint_and_angle(area_from, area_to)

        if self.areas_touch(area_from, area_to) or self.areas_intersect(area_from, area_to):
            geom_from = self.get_area_geometry(area_from)
            geom_to = self.get_area_geometry(area_to)
            border_line = self._get_continuous_border_line(geom_from, geom_to)
            areas_touch = True
        else:
            border_line = self.get_straight_line_between_areas(area_from, area_to)
            areas_touch = False

        return {
            self.PROJECTION_POINT_IDENTIFIER: midpoint,
            self.AZIMUTH_ANGLE_IDENTIFIER: angle,
            self.BORDER_LINE_STRING_IDENTIFIER: border_line,
            self.BORDER_AREAS_TOUCH_IDENTIFIER: areas_touch,
        }

    def areas_touch(self, area_from: str, area_to: str) -> bool:
        """Check if two areas share a common physical (geographic) border.

        Uses Shapely's touches() method to determine if area boundaries
        intersect without overlapping. This is the standard definition
        of physical adjacency for energy market regions.

        Args:
            area_from: Source area identifier
            area_to: Target area identifier

        Returns:
            bool: True if areas share a common boundary, False otherwise
        """
        geom_from = self.get_area_geometry(area_from)
        geom_to = self.get_area_geometry(area_to)
        return geom_from.touches(geom_to)

    def areas_intersect(self, area_from: str, area_to: str) -> bool:
        """Check if two areas have any geometric intersection.

        Uses Shapely's intersects() method to check for any form of geometric
        intersection, including touching, overlapping, or containment. This is
        broader than the touches() check and handles edge cases in geographic data.

        Args:
            area_from: Source area identifier
            area_to: Target area identifier

        Returns:
            bool: True if areas have any geometric intersection, False otherwise

        Note:
            This method is used as a fallback for areas_touch() to handle
            geographic data with small overlaps or slight topology inconsistencies
            that are common in real-world boundary datasets.
        """
        geom_from = self.get_area_geometry(area_from)
        geom_to = self.get_area_geometry(area_to)
        return geom_from.intersects(geom_to)

    def get_area_border_midpoint_and_angle(self, area_from: str, area_to: str) -> tuple[Point, float]:
        """Calculate representative point and directional angle for border.

        The point and angle place directional indicators such as arrows and
        labels.

        For a physical border, the point is the midpoint of the shared
        boundary line, and the angle is perpendicular to the boundary, toward
        the target area. For a logical border, the point is the midpoint of
        the connection line, and the angle follows the connection from source
        to target.

        Args:
            area_from: Source area identifier
            area_to: Target area identifier

        Returns:
            tuple[Point, float]: Representative point and directional angle in degrees.
                Angle range: 0-360 degrees, where 0° is North, 90° is East.
        """
        geom_from = self.get_area_geometry(area_from)
        geom_to = self.get_area_geometry(area_to)

        if self.areas_touch(area_from, area_to) or self.areas_intersect(area_from, area_to):
            midpoint, angle = self._get_midpoint_and_angle_for_touching_areas(geom_from, geom_to)
        else:
            straight_line = self.get_straight_line_between_areas(area_from, area_to)
            midpoint, angle = self._get_midpoint_and_angle_from_line(straight_line)

        # Ensure angle points from area_from to area_to
        if not self._angle_points_to_target(geom_from, geom_to, midpoint, angle):
            angle = (angle + 180) % 360

        return midpoint, angle

    def get_area_geometry(self, area: str) -> Union[Polygon, MultiPolygon]:
        """Retrieve and validate geometry for a specified area.

        Args:
            area: Area identifier that must exist in area_model_gdf index

        Returns:
            Union[Polygon, MultiPolygon]: Cleaned geometry with buffer(0) applied
                to ensure validity for geometric operations

        Raises:
            KeyError: If area is not found in area_model_gdf

        Note:
            The buffer(0) operation ensures geometric validity for complex
            calculations, which is essential for reliable border analysis.
        """
        return self.area_model_gdf.loc[area].geometry.buffer(0)

    def get_straight_line_between_areas(self, area_from: str, area_to: str) -> LineString:
        """Return the line of the logical border between two areas that do not touch.

        The line runs on the line between the representative points of the
        largest polygons of the two areas, from boundary to boundary. If it
        crosses other areas, the line of `NonCrossingPathFinder` replaces it
        when the finder finds one; the finder states its rule. The results
        are cached per pair of areas; the line of B to A is the reversed line
        of A to B.

        Args:
            area_from: Source area identifier
            area_to: Target area identifier

        Returns:
            LineString: Connection line between the area boundaries.
                Line endpoints touch the area boundaries, not the centroids.

        Raises:
            ValueError: If areas are touching (should use physical border instead)
        """
        key = tuple(sorted((area_from, area_to)))
        if key not in self._line_cache:
            self._line_cache[key] = self._compute_straight_line(*key)
        line = self._line_cache[key]
        return line if (area_from, area_to) == key else LineString(line.coords[::-1])

    def _compute_straight_line(self, area_from: str, area_to: str) -> LineString:
        if self.areas_touch(area_from, area_to):
            raise ValueError(f'Areas {area_from} and {area_to} touch - use border line instead')

        geom_from = self._get_largest_polygon(self.get_area_geometry(area_from))
        geom_to = self._get_largest_polygon(self.get_area_geometry(area_to))

        centroid_from = self.get_representative_area_point(geom_from)
        centroid_to = self.get_representative_area_point(geom_to)

        line_full = LineString([centroid_from, centroid_to])

        # Find intersection points with area boundaries
        intersection_from = self._get_boundary_intersection(geom_from, line_full, centroid_to)
        intersection_to = self._get_boundary_intersection(geom_to, line_full, centroid_from)

        straight_line = LineString([intersection_from, intersection_to])

        # Check if line crosses other areas
        if self._line_crosses_other_areas(straight_line, area_from, area_to):
            # Try to find alternative path
            better_line = self._find_non_crossing_line(area_from, area_to)
            if better_line is not None:
                straight_line = better_line
        return straight_line

    def _get_midpoint_and_angle_for_touching_areas(
        self, geom_from: Union[Polygon, MultiPolygon], geom_to: Union[Polygon, MultiPolygon]
    ) -> tuple[Point, float]:
        """Calculate midpoint and angle for physically adjacent areas.

        For areas that share a physical boundary: the midpoint of the shared
        border line and the angle perpendicular to it.

        Args:
            geom_from: Source area geometry
            geom_to: Target area geometry

        Returns:
            tuple[Point, float]: Midpoint of shared border and perpendicular angle
                pointing from source toward target area

        Algorithm:
            1. Extract continuous border line from geometric intersection
            2. Find midpoint along border line (50% interpolation)
            3. Calculate border bearing and perpendicular angle
            4. Ensure angle points from source to target area
        """
        border_line = self._get_continuous_border_line(geom_from, geom_to)
        midpoint = border_line.interpolate(0.5, normalized=True)

        # Get angle perpendicular to border
        start_to_end = self._get_straight_line_from_endpoints(border_line)
        border_bearing = self._calculate_bearing(start_to_end)
        perpendicular_angle = (border_bearing + 90) % 360

        return midpoint, perpendicular_angle

    def _get_midpoint_and_angle_from_line(self, line: LineString) -> tuple[Point, float]:
        """Extract midpoint and directional angle from a LineString.

        Args:
            line: Input LineString geometry

        Returns:
            tuple[Point, float]: Midpoint and bearing angle in degrees

        Note:
            Uses 50% interpolation to find the midpoint, ensuring consistent
            positioning regardless of coordinate density along the line.
        """
        midpoint = line.interpolate(0.5, normalized=True)
        angle = self._calculate_bearing(line)
        return midpoint, angle

    def _angle_points_to_target(
        self,
        geom_from: Union[Polygon, MultiPolygon],
        geom_to: Union[Polygon, MultiPolygon],
        midpoint: Point,
        angle: float,
    ) -> bool:
        """Validate that computed angle points from source toward target area.

        Ensures directional consistency by checking if the angle points closer
        to the target area than to the source area. This is essential for
        correct arrow orientation in energy flow visualization.

        Args:
            geom_from: Source area geometry
            geom_to: Target area geometry
            midpoint: Reference point for angle measurement
            angle: Angle to validate (in degrees)

        Returns:
            bool: True if angle points toward target, False if it points toward source

        Algorithm:
            1. Calculate bearings from midpoint to both area centroids
            2. Compute angular differences between proposed angle and both bearings
            3. Return True if angle is closer to target bearing than source bearing
        """
        centroid_from = self.get_representative_area_point(geom_from)
        centroid_to = self.get_representative_area_point(geom_to)

        bearing_to_from = self._calculate_bearing(LineString([midpoint, centroid_from]))
        bearing_to_to = self._calculate_bearing(LineString([midpoint, centroid_to]))

        angle_diff_from = self._angular_difference(bearing_to_from, angle)
        angle_diff_to = self._angular_difference(bearing_to_to, angle)

        return angle_diff_to < angle_diff_from

    def _get_continuous_border_line(
        self, geom_a: Union[Polygon, MultiPolygon], geom_b: Union[Polygon, MultiPolygon]
    ) -> LineString:
        """Extract shared boundary line between touching geometries.

        Computes the geometric intersection between two touching areas and
        converts the result into a continuous LineString representing the
        shared border. Handles complex intersection geometries including
        multiple segments and mixed geometry types.

        Args:
            geom_a: First area geometry
            geom_b: Second area geometry

        Returns:
            LineString: Continuous line representing the shared boundary

        Raises:
            ValueError: If geometries don't touch or intersect
            TypeError: If intersection cannot be converted to LineString

        Algorithm:
            1. Compute geometric intersection of the two areas
            2. Handle GeometryCollection by extracting line components
            3. Convert Polygon boundaries to LineString if needed
            4. Merge multiple LineStrings into continuous representation
            5. Handle MultiLineString by connecting segments optimally

        Note:
            This method handles the complexity of real-world geographic boundaries
            which may result in complex intersection geometries.
        """
        if not (geom_a.touches(geom_b) or geom_a.intersects(geom_b)):
            raise ValueError('Geometries do not touch or intersect')

        border = geom_a.intersection(geom_b)

        if isinstance(border, GeometryCollection):
            extracted_lines = []

            for g in border.geoms:
                if isinstance(g, LineString):
                    extracted_lines.append(g)
                elif isinstance(g, Polygon):
                    extracted_lines.append(g.boundary)
                elif isinstance(g, MultiLineString):
                    extracted_lines.extend(g.geoms)
                elif isinstance(g, MultiPolygon):
                    extracted_lines.extend([p.boundary for p in g.geoms])

            if not extracted_lines:
                raise TypeError(f'GeometryCollection could not be converted into line: {type(border)}')

            border = linemerge(extracted_lines)

        if isinstance(border, MultiPolygon):
            border = linemerge([p.boundary for p in border.geoms])

        if isinstance(border, Polygon):
            border = border.boundary

        if isinstance(border, MultiLineString):
            border = self._merge_multilinestring(border)

        if isinstance(border, LineString):
            return border

        raise TypeError(f'Unexpected border type: {type(border)}')

    def _get_boundary_intersection(self, geom: Polygon, line: LineString, target_point: Point) -> Point:
        """Return the boundary intersection closest to `target_point`.

        Args:
            geom: Polygon whose boundary to intersect with
            line: LineString to intersect with polygon boundary
            target_point: Reference point for choosing among multiple intersections

        Returns:
            Point: Intersection point closest to target_point

        Raises:
            TypeError: If intersection geometry type is unexpected

        Algorithm:
            1. Compute intersection between line and polygon boundary
            2. Handle different intersection geometry types (Point, MultiPoint, LineString)
            3. For multiple options, select point closest to target
            4. Extract coordinates and create Point geometry
        """
        intersection = geom.boundary.intersection(line)

        if isinstance(intersection, Point):
            return intersection
        elif isinstance(intersection, MultiPoint):
            return min(intersection.geoms, key=lambda p: p.distance(target_point))
        elif isinstance(intersection, (LineString, MultiLineString)):
            # Get all coordinates and find closest
            coords = []
            if isinstance(intersection, LineString):
                coords = list(intersection.coords)
            else:
                for line in intersection.geoms:
                    coords.extend(list(line.coords))
            return Point(min(coords, key=lambda c: Point(c).distance(target_point)))
        else:
            raise TypeError(f'Unexpected intersection type: {type(intersection)}')

    def _line_crosses_other_areas(self, line: LineString, *exclude_areas: str) -> bool:
        """Check if a line crosses through any areas except specified exclusions.

        This method is crucial for validating logical border connections to ensure
        they don't inappropriately cross through other energy market areas, which
        would be misleading in visualization.

        Args:
            line: LineString to test for crossings
            *exclude_areas: Area identifiers to exclude from crossing check
                (typically the source and target areas of the line)

        Returns:
            bool: True if line crosses any non-excluded areas, False otherwise
        """
        other_areas = self.area_model_gdf.drop(list(exclude_areas))
        return other_areas.geometry.crosses(line).any()

    def _find_non_crossing_line(self, area_from: str, area_to: str) -> Union[LineString, None]:
        """Shortest line between the largest polygons of two areas that does not cross other areas.

        Uses the `NonCrossingPathFinder` of the calculator.

        Args:
            area_from: Source area identifier
            area_to: Target area identifier

        Returns:
            LineString or None: Non-crossing line, or None if the path finder
                finds none
        """
        poly_from = self._get_largest_polygon(self.get_area_geometry(area_from))
        poly_to = self._get_largest_polygon(self.get_area_geometry(area_to))

        return self.non_crossing_path_finder.find_shortest_path(
            poly_from, poly_to, self.area_model_gdf.drop([area_from, area_to]), f'{area_from} to {area_to}'
        )

    def _get_largest_polygon(self, geom: Union[Polygon, MultiPolygon]) -> Polygon:
        """Extract largest polygon component from MultiPolygon geometry.

        For MultiPolygon geometries (e.g., countries with islands), this method
        returns the largest polygon by area, which is typically the main landmass.
        This simplifies calculations while focusing on the most significant
        geographic component.

        Args:
            geom: Input geometry (Polygon returned as-is, MultiPolygon simplified)

        Returns:
            Polygon: Largest polygon component by area
        """
        if isinstance(geom, Polygon):
            return geom
        return max(geom.geoms, key=lambda p: p.area)

    def _merge_multilinestring(self, mls: MultiLineString) -> LineString:
        """Merge disconnected LineString segments into continuous line.

        Merges the segments with `shapely.ops.linemerge`. Segments that stay
        apart are joined pairwise, closest pair first, in the orientation that
        gives the shortest line.

        Args:
            mls: MultiLineString with potentially disconnected segments

        Returns:
            LineString: Continuous line through all segments
        """
        merged = linemerge(list(mls.geoms))

        if isinstance(merged, LineString):
            return merged

        # Connect disconnected segments
        lines = list(merged.geoms)

        while len(lines) > 1:
            # Find closest pair
            min_dist = float('inf')
            closest_pair = None

            for i, line1 in enumerate(lines):
                for j, line2 in enumerate(lines[i + 1 :], i + 1):
                    p1, p2 = nearest_points(line1, line2)
                    dist = p1.distance(p2)
                    if dist < min_dist:
                        min_dist = dist
                        closest_pair = (i, j)

            # Connect closest pair
            i, j = closest_pair
            line1, line2 = lines[i], lines[j]

            # Create connected line
            coords1 = list(line1.coords)
            coords2 = list(line2.coords)

            # Find best connection direction
            connections = [
                (coords1 + coords2, LineString(coords1 + coords2).length),
                (coords1 + coords2[::-1], LineString(coords1 + coords2[::-1]).length),
                (coords1[::-1] + coords2, LineString(coords1[::-1] + coords2).length),
                (coords1[::-1] + coords2[::-1], LineString(coords1[::-1] + coords2[::-1]).length),
            ]

            best_coords = min(connections, key=lambda x: x[1])[0]
            new_line = LineString(best_coords)

            # Update lines list
            lines = [l for k, l in enumerate(lines) if k not in (i, j)]
            lines.append(new_line)

        return lines[0]

    def _get_straight_line_from_endpoints(self, line: LineString) -> LineString:
        """Create straight line connecting first and last coordinates.

        Args:
            line: Input LineString with potentially complex path

        Returns:
            LineString: Simplified straight line from start to end

        Use Case:
            Useful for computing bearing angles from complex border geometries
            by simplifying to the fundamental start-to-end direction.
        """
        return LineString([line.coords[0], line.coords[-1]])

    def _calculate_bearing(self, line: LineString) -> float:
        """Calculate compass bearing (Azimuth angle) for a LineString.

        Args:
            line: LineString from which to calculate bearing

        Returns:
            float: Compass bearing (Azimuth angle) in degrees (0-360)
                - 0° = North
                - 90° = East
                - 180° = South
                - 270° = West

        Algorithm:
            Uses the forward azimuth formula from geodetic calculations:
            1. Convert coordinates to radians
            2. Apply spherical trigonometry formulas
            3. Convert result to compass bearing (0-360°)

        Note:
            The bearing is computed on lon/lat coordinates. If the areas have
            a projected CRS, the end points are first transformed to EPSG:4326.
        """
        start = line.coords[0]
        end = line.coords[-1]
        if self._to_lon_lat is not None:
            start, end = self._to_lon_lat.transform(*start[:2]), self._to_lon_lat.transform(*end[:2])

        lat1 = math.radians(start[1])
        lat2 = math.radians(end[1])

        diff_lon = math.radians(end[0] - start[0])

        x = math.sin(diff_lon) * math.cos(lat2)
        y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(diff_lon)

        bearing = math.atan2(x, y)
        bearing = math.degrees(bearing)
        compass_bearing = (bearing + 360) % 360

        return compass_bearing

    def _angular_difference(self, angle1: float, angle2: float) -> float:
        """Calculate minimum angular difference between two compass bearings.

        Handles the circular nature of compass bearings to find the shortest
        angular distance between two directions, accounting for the 0°/360° wraparound.

        Args:
            angle1: First angle in degrees (0-360)
            angle2: Second angle in degrees (0-360)

        Returns:
            float: Minimum angular difference in degrees (0-180)
        """
        diff = abs(angle1 - angle2) % 360
        return min(diff, 360 - diff)


class NonCrossingPathFinder:
    """Shortest line between two polygons that does not cross the buffered other areas.

    The finder samples points on the boundaries of both polygons and tests
    every pair of points. It buffers the other areas by `min_clearance` and
    rejects a line that crosses the boundary of a buffered area. A line that
    lies wholly inside a buffer is accepted, and it can cross that area. The
    number of tested lines is `num_points` squared.

    Attributes:
        num_points (int): Number of boundary points to sample per area
        min_clearance (float): Buffer distance around other areas, in EPSG:3857
            (Web Mercator) units if the areas have a geographic CRS, where one
            unit is about cos(latitude) meters on the ground, else in the units
            of their CRS.
        show_progress (bool): Whether to display progress bars for long operations

    Examples:
        >>> import geopandas as gpd
        >>> from shapely.geometry import box
        >>> finder = NonCrossingPathFinder(num_points=20, min_clearance=10_000, show_progress=False)
        >>> obstacle = gpd.GeoDataFrame(geometry=[box(1.5, -1, 2.5, 0.5)], crs='EPSG:4326')
        >>> path = finder.find_shortest_path(box(0, 0, 1, 1), box(3, 0, 4, 1), obstacle)
        >>> [round(value, 1) for value in path.bounds]  # the line passes above the obstacle
        [1.0, 0.6, 3.0, 0.6]
    """

    def __init__(self, num_points: int = 100, min_clearance: float = 50000, show_progress: bool = True):
        """Initialize the non-crossing path finder.

        Args:
            num_points: Number of boundary points to sample per polygon.
                Higher values improve path quality but increase computation time.
                Typical range: 50-500 depending on precision requirements.
            min_clearance: Buffer distance around other areas, in EPSG:3857
                (Web Mercator) units if the areas have a geographic CRS, where
                one unit is about cos(latitude) meters on the ground, else in
                the units of their CRS.
            show_progress: Whether to display progress bars during computation.
                Useful for long-running operations with high num_points values.
        """
        self.num_points = num_points
        self.min_clearance = min_clearance
        self.show_progress = show_progress

    def find_shortest_path(
        self, polygon1: Polygon, polygon2: Polygon, other_areas: gpd.GeoDataFrame, name: str = None
    ) -> Union[LineString, None]:
        """Return the shortest line between two polygons that the finder accepts.

        The run time grows with the number of tested lines times the number
        of other areas.

        Args:
            polygon1: Source polygon geometry
            polygon2: Target polygon geometry
            other_areas: GeoDataFrame of the areas to buffer and not cross
            name: Optional name for progress tracking display

        Returns:
            LineString or None: Shortest accepted line, or None if no line is
                accepted

        Raises:
            ValueError: If `other_areas` has no CRS.
        """
        buffered_areas = self._buffer_areas(other_areas, self.min_clearance)
        points1 = self._get_boundary_points(polygon1, self.num_points)
        points2 = self._get_boundary_points(polygon2, self.num_points)

        lines = [LineString([p1, p2]) for p1, p2 in itertools.product(points1, points2)]
        shortest_line = None
        min_length = float('inf')

        iterator = tqdm(lines, desc=f'Finding path for {name or "path"}') if self.show_progress else lines

        for line in iterator:
            if not buffered_areas.geometry.crosses(line).any():
                if line.length < min_length:
                    shortest_line = line
                    min_length = line.length

        return shortest_line

    def _buffer_areas(self, areas: gpd.GeoDataFrame, buffer_distance: float) -> gpd.GeoDataFrame:
        """Return the areas buffered by `buffer_distance`, in their original CRS.

        Args:
            areas: GeoDataFrame containing areas to buffer
            buffer_distance: Buffer distance, in EPSG:3857 (Web Mercator)
                units if the areas have a geographic CRS, where one unit is
                about cos(latitude) meters on the ground, else in the units of
                their CRS.

        Returns:
            gpd.GeoDataFrame: Areas with buffered geometries in original CRS

        Raises:
            ValueError: If `areas` has no CRS.
        """
        areas_copy = areas.copy()
        original_crs = areas_copy.crs

        if original_crs is None:
            raise ValueError('GeoDataFrame must have a valid CRS defined.')

        if original_crs.is_geographic:
            # Use Web Mercator for accurate distance-based buffering
            projected_crs = 'EPSG:3857'
            areas_copy = areas_copy.to_crs(projected_crs)
            areas_copy['geometry'] = areas_copy.buffer(buffer_distance)
            areas_copy = areas_copy.to_crs(original_crs)
        else:
            # Already in projected coordinates
            areas_copy['geometry'] = areas_copy.buffer(buffer_distance)

        return areas_copy

    def _get_boundary_points(self, polygon: Polygon, num_points: int) -> list[Point]:
        """Sample evenly distributed points along polygon boundary.

        Creates a uniform sampling of points along the polygon perimeter using
        interpolation. The points are the candidates of the path finder.

        Args:
            polygon: Input polygon to sample
            num_points: Number of points to sample along boundary

        Returns:
            list[Point]: List of evenly spaced boundary points

        Algorithm:
            1. Calculate total boundary length
            2. Divide into equal segments
            3. Interpolate points at regular intervals
            4. Return as list of Point geometries

        Note:
            Points are distributed proportionally to boundary length,
            ensuring uniform density regardless of polygon complexity.
        """
        boundary = polygon.boundary
        total_length = boundary.length
        # Generate evenly spaced points along boundary
        return [boundary.interpolate((i / num_points) * total_length) for i in range(num_points)]
