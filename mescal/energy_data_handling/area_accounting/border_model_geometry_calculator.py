from typing import Tuple, Union, List
import math
import itertools
from tqdm import tqdm
import geopandas as gpd
from shapely.geometry import Point, MultiPoint, LineString, MultiLineString, Polygon, MultiPolygon, GeometryCollection
from shapely.ops import nearest_points, linemerge

from mescal.energy_data_handling.area_accounting.model_generator_base import GeoModelGeneratorBase


class AreaBorderGeometryCalculator(GeoModelGeneratorBase):
    """Advanced geometric calculator for energy system area border analysis.
    
    This class provides sophisticated geometric calculations for borders between
    energy system areas, handling both physical borders (adjacent areas sharing
    geographic boundaries) and logical borders (non-adjacent areas requiring
    connection paths, e.g. through the sea). It's specifically designed to generate
    properties for energy market cross-border visualizations.
    
    The calculator combines multiple geometric algorithms:
    - Physical border extraction using geometric intersection
    - Logical geo-line-border path finding with obstacle avoidance
    - Representative point computation using pole of inaccessibility for label placements on maps
    - Azimuth angle calculation for flow icon (arrow) visualization
    - Geometric validation and optimization
    
    Key Features:
        - Automatic detection of physical vs logical borders
        - Optimal path finding for non-crossing connections
        - Representative point calculation for label placement
        - Directional angle computation for arrow orientation
        - Performance optimization with geometric caching
        - Integration with MESCAL area accounting workflows
    
    Energy Domain Applications:
        - Visualization of cross-border (cross-country, cross-biddingzone, cross-macroregion) variables (flows, spreads, capacities, ...)
    
    Attributes:
        area_model_gdf (gpd.GeoDataFrame): GeoDataFrame with area polygon geometries
        non_crossing_path_finder (NonCrossingPathFinder): Path optimization engine
        _centroid_cache (dict): Cached representative points for performance
        _line_cache (dict): Cached border lines for repeated calculations
    
    Example:
        >>> import geopandas as gpd
        >>> from shapely.geometry import box
        >>> 
        >>> # Setup area geometries
        >>> areas = gpd.GeoDataFrame({
        ...     'geometry': [box(0, 0, 1, 1), box(2, 0, 3, 1)]  # Two separate areas
        ... }, index=['Area_A', 'Area_B'])
        >>> 
        >>> # Calculate border geometry
        >>> calculator = AreaBorderGeometryCalculator(areas)
        >>> border_info = calculator.calculate_border_geometry('Area_A', 'Area_B')
        >>> print(f"Border type: {'Physical' if border_info['is_physical'] else 'Logical'}")
    """

    PROJECTION_POINT_IDENTIFIER = 'projection_point'
    AZIMUTH_ANGLE_IDENTIFIER = 'azimuth_angle'
    BORDER_IS_PHYSICAL_IDENTIFIER = 'is_physical'
    BORDER_LINE_STRING_IDENTIFIER = 'geo_line_string'
    
    def __init__(self, area_model_gdf: gpd.GeoDataFrame, non_crossing_path_finder: 'NonCrossingPathFinder' = None):
        """Initialize the border geometry calculator.
        
        Args:
            area_model_gdf: GeoDataFrame containing area geometries with polygon
                boundaries. Index should contain area identifiers (e.g., country codes,
                bidding zone names). Must contain valid polygon geometries in 'geometry' column.
            non_crossing_path_finder: Optional custom path finder for logical borders.
                If None, creates default NonCrossingPathFinder with standard parameters.
                
        Raises:
            ValueError: If geometries are invalid or area_model_gdf lacks required structure
            
        Example:
            >>> areas_gdf = gpd.read_file('countries.geojson').set_index('ISO_A2')
            >>> calculator = AreaBorderGeometryCalculator(areas_gdf)
            >>> 
            >>> # Custom path finder for specific requirements
            >>> custom_finder = NonCrossingPathFinder(num_points=200, min_clearance=10000)
            >>> calculator = AreaBorderGeometryCalculator(areas_gdf, custom_finder)
            
        Note:
            Invalid geometries are automatically cleaned using buffer(0) operation.
            Large area datasets benefit from using projected coordinate systems
            for accurate geometric calculations.
        """
        self.area_model_gdf = area_model_gdf
        self.non_crossing_path_finder = non_crossing_path_finder or NonCrossingPathFinder()
        self._validate_geometries()

        self._centroid_cache: dict[str, Point] = {}
        self._line_cache: dict[Tuple[str, str], LineString] = {}
    
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
        self, 
        area_from: str, 
        area_to: str
    ) -> dict[str, Union[Point, float, LineString, bool]]:
        """Calculate comprehensive geometric properties for an area border.
        
        This is the main interface method that computes all geometric properties
        needed for border visualization and analysis. It automatically detects
        whether areas are physically adjacent or logically connected and applies
        appropriate geometric algorithms.
        
        Processing Logic:
            1. Detect if areas share physical boundary (touching/intersecting)
            2. For physical borders: extract shared boundary line
            3. For logical borders: compute optimal connection path
            4. Calculate representative point for label/arrow placement
            5. Compute azimuth angle for arrow icon visualization
        
        Args:
            area_from: Source area identifier (must exist in area_model_gdf index)
            area_to: Target area identifier (must exist in area_model_gdf index)
            
        Returns:
            dict: Comprehensive border geometry information containing:
                - 'projection_point' (Point): Optimal point for label/arrow placement
                - 'azimuth_angle' (float): Directional angle in degrees (0-360)
                - 'geo_line_string' (LineString): Border line geometry
                - 'is_physical' (bool): True for touching areas, False for logical borders
                
        Raises:
            KeyError: If area_from or area_to not found in area_model_gdf
            ValueError: If geometric calculations fail
            
        Example:
            >>> border_info = calculator.calculate_border_geometry('DE', 'FR')
            >>> 
            >>> # Use for visualization
            >>> point = border_info['projection_point']
            >>> angle = border_info['azimuth_angle']
            >>> is_physical = border_info['is_physical']
            >>> 
            >>> print(f"Border DE→FR: {point} at {angle}° ({'physical' if is_physical else 'logical'})")
        """
        midpoint, angle = self.get_area_border_midpoint_and_angle(area_from, area_to)
        
        if self.areas_touch(area_from, area_to) or self.areas_intersect(area_from, area_to):
            geom_from = self.get_area_geometry(area_from)
            geom_to = self.get_area_geometry(area_to)
            border_line = self._get_continuous_border_line(geom_from, geom_to)
            is_physical = True
        else:
            border_line = self.get_straight_line_between_areas(area_from, area_to)
            is_physical = False
        
        return {
            self.PROJECTION_POINT_IDENTIFIER: midpoint,
            self.AZIMUTH_ANGLE_IDENTIFIER: angle,
            self.BORDER_LINE_STRING_IDENTIFIER: border_line,
            self.BORDER_IS_PHYSICAL_IDENTIFIER: is_physical
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
            
        Example:
            >>> touching = calculator.areas_touch('DE', 'FR')  # True for neighboring countries
            >>> separated = calculator.areas_touch('DE', 'GB')  # False for non-adjacent countries
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
    
    def get_area_border_midpoint_and_angle(
        self, 
        area_from: str, 
        area_to: str
    ) -> tuple[Point, float]:
        """Calculate representative point and directional angle for border.
        
        Computes the optimal point for placing directional indicators (arrows,
        labels) and the corresponding angle for proper orientation. The algorithm
        adapts to both physical and logical borders to ensure optimal placement.
        
        For Physical Borders:
            - Uses midpoint of shared boundary line
            - Angle is perpendicular to boundary, pointing toward target area
        
        For Logical Borders:
            - Uses midpoint of optimal connection line
            - Angle follows connection direction from source to target
        
        Args:
            area_from: Source area identifier
            area_to: Target area identifier
            
        Returns:
            tuple[Point, float]: Representative point and directional angle in degrees.
                Angle range: 0-360 degrees, where 0° is North, 90° is East.
                
        Example:
            >>> point, angle = calculator.get_area_border_midpoint_and_angle('DE', 'FR')
            >>> print(f"Place arrow at {point} oriented at {angle}° for DE→FR flow")
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
        """Compute optimal straight-line connection between non-adjacent areas.
        
        Creates a direct line connection between area boundaries, with intelligent
        path optimization to avoid crossing other areas when possible. This is
        particularly important for non-physical borders.
        
        Algorithm:
            1. Find representative points for both areas
            2. Create line connecting area centroids
            3. Calculate intersection points with area boundaries  
            4. Check for conflicts with other areas
            5. Apply non-crossing path optimization if needed
        
        Args:
            area_from: Source area identifier
            area_to: Target area identifier
            
        Returns:
            LineString: Optimized connection line between area boundaries.
                Line endpoints touch the area boundaries, not the centroids.
                
        Raises:
            ValueError: If areas are touching (should use physical border instead)
            
        Example:
            >>> # Connect non-adjacent areas (e.g., Germany to UK)
            >>> line = calculator.get_straight_line_between_areas('DE', 'GB')
            >>> print(f"Connection length: {line.length:.0f} km")
            
        Performance Note:
            Results are cached to improve performance for repeated calculations.
            Path optimization can be computationally intensive for complex geometries.
        """
        key = tuple(sorted((area_from, area_to)))
        if key in self._line_cache:
            return self._line_cache[key]

        if self.areas_touch(area_from, area_to):
            raise ValueError(f"Areas {area_from} and {area_to} touch - use border line instead")
        
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

        self._line_cache[key] = straight_line
        return straight_line
    
    def _get_midpoint_and_angle_for_touching_areas(
        self,
        geom_from: Union[Polygon, MultiPolygon],
        geom_to: Union[Polygon, MultiPolygon]
    ) -> tuple[Point, float]:
        """Calculate midpoint and angle for physically adjacent areas.
        
        For areas that share a physical boundary, this method extracts the
        shared border line and computes the optimal point and angle for
        directional indicators.
        
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
        angle: float
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
        self,
        geom_a: Union[Polygon, MultiPolygon],
        geom_b: Union[Polygon, MultiPolygon]
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
        """Get the shared border between two touching geometries."""
        if not (geom_a.touches(geom_b) or geom_a.intersects(geom_b)):
            raise ValueError("Geometries do not touch or intersect")
        
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
                raise TypeError(f"GeometryCollection could not be converted into line: {type(border)}")

            border = linemerge(extracted_lines)

        if isinstance(border, MultiPolygon):
            border = linemerge([p.boundary for p in border.geoms])

        if isinstance(border, Polygon):
            border = border.boundary

        if isinstance(border, MultiLineString):
            border = self._merge_multilinestring(border)

        if isinstance(border, LineString):
            return border

        raise TypeError(f"Unexpected border type: {type(border)}")
    
    def _get_boundary_intersection(
        self,
        geom: Polygon,
        line: LineString,
        target_point: Point
    ) -> Point:
        """Find optimal intersection point between line and polygon boundary.
        
        When a line intersects a polygon boundary at multiple points, this method
        selects the point that is closest to a specified target point. This is
        essential for creating clean border connections.
        
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
        """Find where a line intersects a polygon boundary, choosing the point closest to target."""
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
            raise TypeError(f"Unexpected intersection type: {type(intersection)}")
    
    def _line_crosses_other_areas(
        self,
        line: LineString,
        *exclude_areas: str
    ) -> bool:
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
            
        Energy Domain Context:
            In energy market visualization, border lines should represent direct
            connections without implying transit through intermediate areas unless
            explicitly modeled.
        """
        """Check if a line crosses any areas except the excluded ones."""
        other_areas = self.area_model_gdf.drop(list(exclude_areas))
        return other_areas.geometry.crosses(line).any()
    
    def _find_non_crossing_line(
        self,
        area_from: str,
        area_to: str
    ) -> Union[LineString, None]:
        """Find optimal connection path that avoids crossing other areas.
        
        Uses the NonCrossingPathFinder to compute the shortest connection between
        two areas that maintains minimum clearance from other areas. This creates
        clean visualization paths for logical borders.
        
        Args:
            area_from: Source area identifier
            area_to: Target area identifier
            
        Returns:
            LineString or None: Optimal non-crossing path, or None if no suitable
                path found within the configured constraints
                
        Algorithm:
            1. Extract largest polygons from MultiPolygon geometries
            2. Create exclusion set of all other areas
            3. Apply NonCrossingPathFinder algorithm
            4. Return shortest valid path or None if impossible
            
        Performance Note:
            This operation can be computationally intensive for complex geometries
            and large numbers of areas. Results are cached for efficiency.
        """
        poly_from = self._get_largest_polygon(self.get_area_geometry(area_from))
        poly_to = self._get_largest_polygon(self.get_area_geometry(area_to))

        return self.non_crossing_path_finder.find_shortest_path(
            poly_from,
            poly_to,
            self.area_model_gdf.drop([area_from, area_to]),
            f"{area_from} to {area_to}"
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
        
        Converts a MultiLineString into a single continuous LineString by
        intelligently connecting segments to minimize total length while
        preserving the overall geometric relationship.
        
        Args:
            mls: MultiLineString with potentially disconnected segments
            
        Returns:
            LineString: Continuous line connecting all segments optimally
            
        Algorithm:
            1. Attempt automatic merge using Shapely's linemerge
            2. If segments remain disconnected, apply iterative connection:
               - Find closest pair of segment endpoints
               - Connect segments with optimal orientation
               - Repeat until single continuous line achieved
            
        Note:
            This method is particularly important for complex international
            borders that may be represented as multiple disconnected segments
            in geographic datasets.
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
                for j, line2 in enumerate(lines[i+1:], i+1):
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
                (coords1[::-1] + coords2[::-1], LineString(coords1[::-1] + coords2[::-1]).length)
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
            This implementation assumes coordinates are in geographic (lat/lon)
            format. For projected coordinates, results approximate true bearings
            within reasonable accuracy for visualization purposes.
        """
        start = line.coords[0]
        end = line.coords[-1]
        
        lat1 = math.radians(start[1])
        lat2 = math.radians(end[1])
        
        diff_lon = math.radians(end[0] - start[0])
        
        x = math.sin(diff_lon) * math.cos(lat2)
        y = (math.cos(lat1) * math.sin(lat2) - 
             math.sin(lat1) * math.cos(lat2) * math.cos(diff_lon))
        
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
            
        Example:
            >>> diff = calculator._angular_difference(10, 350)  # Returns 20, not 340
            >>> diff = calculator._angular_difference(90, 270)  # Returns 180
        """
        diff = abs(angle1 - angle2) % 360
        return min(diff, 360 - diff)


class NonCrossingPathFinder:
    """Optimized path finder for non-crossing connections between areas.
    
    This class implements an algorithm for finding the shortest
    path between two polygon areas while maintaining specified clearance from
    other areas. It's specifically designed for energy system visualization
    where geographic border line representations should not misleadingly
    cross through other market areas.
    
    The algorithm uses a brute-force approach to test multiple potential paths
    and select the optimal solution.
    
    Key Features:
        - Configurable boundary point sampling density
        - Adjustable minimum clearance distances
        - Progress tracking for long-running operations
        - Optimization for common geometric scenarios
    
    Performance Characteristics:
        - Time complexity: O(n² × m) where n=num_points, m=number of other areas
        - Memory usage scales with point sampling density
        - Results improve with higher point sampling but at computational cost
    
    Attributes:
        num_points (int): Number of boundary points to sample per area
        min_clearance (float): Minimum distance from other areas (in CRS units)
        show_progress (bool): Whether to display progress bars for long operations
    
    Example:
        >>> # High-precision path finding
        >>> finder = NonCrossingPathFinder(num_points=500, min_clearance=50000)
        >>> path = finder.find_shortest_path(area1_poly, area2_poly, other_areas_gdf)
    """

    def __init__(
        self,
        num_points: int = 100,
        min_clearance: float = 50000,
        show_progress: bool = True
    ):
        """Initialize the non-crossing path finder.
        
        Args:
            num_points: Number of boundary points to sample per polygon.
                Higher values improve path quality but increase computation time.
                Typical range: 50-500 depending on precision requirements.
            min_clearance: Minimum clearance distance from other areas in
                coordinate reference system units. For geographic coordinates,
                this is typically in meters when using projected CRS.
            show_progress: Whether to display progress bars during computation.
                Useful for long-running operations with high num_points values.
                
        Example:
            >>> # High-precision finder for detailed analysis
            >>> finder = NonCrossingPathFinder(
            ...     num_points=300,      # High sampling density
            ...     min_clearance=25000, # 25km minimum clearance
            ...     show_progress=True   # Show progress for long operations
            ... )
        """
        self.num_points = num_points
        self.min_clearance = min_clearance
        self.show_progress = show_progress

    def find_shortest_path(
        self,
        polygon1: Polygon,
        polygon2: Polygon,
        other_areas: gpd.GeoDataFrame,
        name: str = None
    ) -> Union[LineString, None]:
        """Find shortest non-crossing path between two polygons.
        
        Tests all combinations of boundary points between two polygons to find
        the shortest connection that maintains minimum clearance from other areas.
        If the algorithm succeedes and finds a non-crossing LineString, it ensures
        clean visualization paths for energy market border analysis.
        
        Args:
            polygon1: Source polygon geometry
            polygon2: Target polygon geometry
            other_areas: GeoDataFrame of areas to avoid crossing through
            name: Optional name for progress tracking display
            
        Returns:
            LineString or None: Shortest valid path between polygons, or None
                if no path meets clearance requirements
                
        Algorithm:
            1. Sample boundary points for both polygons
            2. Buffer other areas by minimum clearance distance
            3. Test all point-to-point connections
            4. Filter out paths that cross buffered areas
            5. Return shortest valid path
        
        Performance Scaling:
            - Total paths tested: num_points² 
            - With default num_points=100: tests 10,000 potential paths
            - Computation time scales roughly O(n² × m) where m = number of other areas
            
        Example:
            >>> path = finder.find_shortest_path(
            ...     source_area, target_area, obstacles_gdf, "Germany to UK"
            ... )
            >>> if path:
            ...     print(f"Found path with length: {path.length:.0f} meters")
            ... else:
            ...     print("No valid path found with current clearance settings")
        """
        buffered_areas = self._buffer_areas(other_areas, self.min_clearance)
        points1 = self._get_boundary_points(polygon1, self.num_points)
        points2 = self._get_boundary_points(polygon2, self.num_points)

        lines = [LineString([p1, p2]) for p1, p2 in itertools.product(points1, points2)]
        shortest_line = None
        min_length = float('inf')

        iterator = tqdm(lines, desc=f"Finding path for {name or 'path'}") if self.show_progress else lines

        for line in iterator:
            if not buffered_areas.geometry.crosses(line).any():
                if line.length < min_length:
                    shortest_line = line
                    min_length = line.length

        return shortest_line

    def _buffer_areas(self, areas: gpd.GeoDataFrame, buffer_distance: float) -> gpd.GeoDataFrame:
        """Apply buffer operation to create clearance zones around areas.
        
        Creates expanded geometries around areas to enforce minimum clearance
        distances. Handles coordinate system transformations to ensure accurate
        distance-based buffering regardless of input CRS.
        
        Args:
            areas: GeoDataFrame containing areas to buffer
            buffer_distance: Buffer distance in meters
            
        Returns:
            gpd.GeoDataFrame: Areas with buffered geometries in original CRS
            
        Raises:
            ValueError: If GeoDataFrame lacks valid CRS definition
            
        Algorithm:
            1. Check if CRS is geographic (lat/lon)
            2. If geographic, temporarily project to Web Mercator (EPSG:3857)
            3. Apply buffer operation in projected coordinates
            4. Transform back to original CRS
            5. If already projected, buffer directly
        """
        areas_copy = areas.copy()
        original_crs = areas_copy.crs

        if original_crs is None:
            raise ValueError("GeoDataFrame must have a valid CRS defined.")

        if original_crs.is_geographic:
            # Use Web Mercator for accurate distance-based buffering
            projected_crs = "EPSG:3857"
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
        interpolation. This provides comprehensive coverage for path-finding
        while maintaining computational efficiency.
        
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


if __name__ == '__main__':
    # Comprehensive demonstration of AreaBorderGeometryCalculator capabilities
    import geopandas as gpd
    from shapely.geometry import box, Point
    import matplotlib.pyplot as plt
    import numpy as np
    
    print("=== AREA BORDER GEOMETRY CALCULATOR DEMONSTRATION ===")
    
    # Create synthetic European-style area data for demonstration
    print("\n1. CREATING SYNTHETIC AREA DATA")
    print("=" * 50)
    
    # Define area geometries with both touching and separated regions
    area_geometries = {
        'DE': box(8, 47, 15, 55),           # Germany (approximate bounds)
        'FR': box(0, 42, 8, 51),            # France (touches Germany)
        'GB': box(-8, 50, 2, 59),                   # Great Britain (separated)
        'PL': box(14, 49, 24, 55),          # Poland (touches Germany)
        'ES': box(-9, 36, 4, 44),           # Spain (touches France)
        'IT': box(6, 36, 19, 47),           # Italy (separated from most)
        'NL': box(3, 50, 8, 54),            # Netherlands (touches Germany and France)
    }
    
    # Create GeoDataFrame
    areas_gdf = gpd.GeoDataFrame(
        index=list(area_geometries.keys()),
        geometry=list(area_geometries.values()),
        crs='EPSG:4326'  # WGS84 geographic coordinates
    )
    
    # Add some area properties for context
    areas_gdf['area_type'] = 'country'
    areas_gdf['region'] = ['Central Europe', 'Western Europe', 'British Isles', 
                           'Eastern Europe', 'Southern Europe', 'Southern Europe', 'Western Europe']
    
    print(f"Created {len(areas_gdf)} synthetic areas:")
    for area_id, area_data in areas_gdf.iterrows():
        bounds = area_data.geometry.bounds
        print(f"  {area_id}: {area_data['region']} (bounds: {bounds[0]:.1f}, {bounds[1]:.1f} to {bounds[2]:.1f}, {bounds[3]:.1f})")
    
    # Initialize calculator
    calculator = AreaBorderGeometryCalculator(areas_gdf)
    
    # 2. Test different border types
    print("\n2. BORDER TYPE ANALYSIS")
    print("=" * 50)
    
    test_pairs = [
        ('DE', 'FR'),  # Physical border (touching)
        ('DE', 'PL'),  # Physical border (touching)
        ('DE', 'GB'),  # Logical border (separated)
        ('FR', 'ES'),  # Physical border (touching)
        ('GB', 'IT'),  # Logical border (separated)
        ('NL', 'PL'),  # Logical border (through Germany)
    ]
    
    border_results = []
    
    for area_from, area_to in test_pairs:
        print(f"\nAnalyzing border: {area_from} → {area_to}")
        
        # Check physical relationship
        touches = calculator.areas_touch(area_from, area_to)
        intersects = calculator.areas_intersect(area_from, area_to)
        
        print(f"  Physical relationship: touches={touches}, intersects={intersects}")
        
        # Calculate complete border geometry
        try:
            geometry_info = calculator.calculate_border_geometry(area_from, area_to)
            
            projection_point = geometry_info[calculator.PROJECTION_POINT_IDENTIFIER]
            azimuth_angle = geometry_info[calculator.AZIMUTH_ANGLE_IDENTIFIER]
            is_physical = geometry_info[calculator.BORDER_IS_PHYSICAL_IDENTIFIER]
            border_line = geometry_info[calculator.BORDER_LINE_STRING_IDENTIFIER]
            
            border_type = "Physical" if is_physical else "Logical"
            
            print(f"  Border type: {border_type}")
            print(f"  Projection point: ({projection_point.x:.3f}, {projection_point.y:.3f})")
            print(f"  Azimuth angle: {azimuth_angle:.1f}°")
            print(f"  Border line length: {border_line.length:.3f} degrees")
            
            border_results.append({
                'from_area': area_from,
                'to_area': area_to,
                'border_type': border_type,
                'projection_point': projection_point,
                'azimuth_angle': azimuth_angle,
                'border_line': border_line,
                'line_length': border_line.length
            })
            
        except Exception as e:
            print(f"  Error calculating geometry: {e}")
    
    # 3. Analyze connectivity patterns
    print("\n3. CONNECTIVITY PATTERN ANALYSIS")
    print("=" * 50)
    
    physical_borders = [(r['from_area'], r['to_area']) for r in border_results if r['border_type'] == 'Physical']
    logical_borders = [(r['from_area'], r['to_area']) for r in border_results if r['border_type'] == 'Logical']
    
    print(f"Physical borders ({len(physical_borders)}): {physical_borders}")
    print(f"Logical borders ({len(logical_borders)}): {logical_borders}")
    
    # Find areas that touch each other
    all_areas = list(areas_gdf.index)
    touching_pairs = []
    
    for i, area1 in enumerate(all_areas):
        for area2 in all_areas[i+1:]:
            if calculator.areas_touch(area1, area2):
                touching_pairs.append((area1, area2))
    
    print(f"\nAll physically touching area pairs: {touching_pairs}")
    
    # 4. Demonstrate path optimization
    print("\n4. PATH OPTIMIZATION DEMONSTRATION")
    print("=" * 50)
    
    # Test non-crossing path finder
    path_finder = NonCrossingPathFinder(num_points=50, min_clearance=0.5, show_progress=False)
    
    print("Testing non-crossing path optimization...")
    
    for area_from, area_to in [('GB', 'PL'), ('ES', 'IT')]:
        print(f"\nFinding optimal path: {area_from} → {area_to}")
        
        try:
            geom_from = calculator.get_area_geometry(area_from)
            geom_to = calculator.get_area_geometry(area_to)
            other_areas = areas_gdf.drop([area_from, area_to])
            
            optimal_path = path_finder.find_shortest_path(
                geom_from, geom_to, other_areas, f"{area_from} to {area_to}"
            )
            
            if optimal_path:
                print(f"  Found optimal path with length: {optimal_path.length:.3f} degrees")
                print(f"  Path coordinates: {list(optimal_path.coords)[:2]}...{list(optimal_path.coords)[-2:]}")
            else:
                print(f"  No suitable path found with current clearance settings")
                
        except Exception as e:
            print(f"  Path optimization failed: {e}")
    
    # 5. Performance and caching demonstration
    print("\n5. PERFORMANCE AND CACHING")
    print("=" * 50)
    
    import time
    
    # Test caching performance
    test_border = ('DE', 'FR')
    
    # First calculation (no cache)
    start_time = time.time()
    result1 = calculator.calculate_border_geometry(*test_border)
    first_time = time.time() - start_time
    
    # Second calculation (with cache)
    start_time = time.time()
    result2 = calculator.calculate_border_geometry(*test_border)
    second_time = time.time() - start_time
    
    print(f"Performance comparison for {test_border[0]} → {test_border[1]}:")
    print(f"  First calculation (no cache): {first_time*1000:.2f} ms")
    print(f"  Second calculation (cached): {second_time*1000:.2f} ms")
    print(f"  Speed improvement: {first_time/second_time:.1f}x faster")
    
    # Verify cache consistency
    point1 = result1[calculator.PROJECTION_POINT_IDENTIFIER]
    point2 = result2[calculator.PROJECTION_POINT_IDENTIFIER]
    same_result = abs(point1.x - point2.x) < 1e-10 and abs(point1.y - point2.y) < 1e-10
    print(f"  Cache consistency: {'✓ Passed' if same_result else '✗ Failed'}")
    
    # 6. Summary statistics
    print("\n6. SUMMARY STATISTICS")
    print("=" * 50)
    
    if border_results:
        physical_count = sum(1 for r in border_results if r['border_type'] == 'Physical')
        logical_count = len(border_results) - physical_count
        
        avg_physical_length = np.mean([r['line_length'] for r in border_results if r['border_type'] == 'Physical']) if physical_count > 0 else 0
        avg_logical_length = np.mean([r['line_length'] for r in border_results if r['border_type'] == 'Logical']) if logical_count > 0 else 0
        
        print(f"Border analysis summary:")
        print(f"  Total borders analyzed: {len(border_results)}")
        print(f"  Physical borders: {physical_count} (avg length: {avg_physical_length:.3f}°)")
        print(f"  Logical borders: {logical_count} (avg length: {avg_logical_length:.3f}°)")
        print(f"  Areas with most connections: {max(set([r['from_area'] for r in border_results] + [r['to_area'] for r in border_results]), key=lambda x: sum(1 for r in border_results if r['from_area'] == x or r['to_area'] == x))}")
    
    print("\n=== DEMONSTRATION COMPLETE ===")
    print("\nThis example demonstrates:")
    print("- Physical vs logical border detection and handling")
    print("- Representative point and angle calculation for visualization")
    print("- Non-crossing path optimization for clean border representation")
    print("- Performance optimization through geometric caching")
    print("- Comprehensive border analysis for energy market applications")
    print("- Integration with GeoDataFrame workflows")
    print("- Error handling and validation for real-world geometric data")
    
    # Optional: Create a simple visualization if matplotlib available
    try:
        import matplotlib.patches as patches
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Plot areas
        areas_gdf.plot(ax=ax, alpha=0.7, edgecolor='black')
        
        # Plot border lines and points
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
        for i, result in enumerate(border_results[:6]):
            color = colors[i % len(colors)]
            
            # Plot border line
            line_coords = list(result['border_line'].coords)
            line_x, line_y = zip(*line_coords)
            ax.plot(line_x, line_y, color=color, linewidth=2, 
                   linestyle='-' if result['border_type'] == 'Physical' else '--',
                   label=f"{result['from_area']}→{result['to_area']} ({result['border_type']})")
            
            # Plot projection point
            pt = result['projection_point']
            ax.plot(pt.x, pt.y, 'o', color=color, markersize=6)
        
        # Add area labels
        for area_id, area_data in areas_gdf.iterrows():
            centroid = area_data.geometry.centroid
            ax.text(centroid.x, centroid.y, area_id, fontsize=10, 
                   ha='center', va='center', fontweight='bold')
        
        ax.set_title('Area Border Geometry Analysis\nSolid lines: Physical borders, Dashed lines: Logical borders')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        
        plt.tight_layout()
        plt.savefig('border_geometry_demo.png', dpi=150, bbox_inches='tight')
        print("\nVisualization saved as 'border_geometry_demo.png'")
        
    except ImportError:
        print("\nNote: Install matplotlib for visualization features")
