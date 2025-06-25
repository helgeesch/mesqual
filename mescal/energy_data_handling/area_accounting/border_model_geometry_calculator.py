from typing import Tuple, Union, List
import math
import itertools
from tqdm import tqdm
import geopandas as gpd
from shapely.geometry import Point, MultiPoint, LineString, MultiLineString, Polygon, MultiPolygon
from shapely.ops import nearest_points, linemerge


class AreaBorderGeometryCalculator:
    """Calculates geometric properties for area borders including midpoints, angles, and line representations.
    
    This class provides sophisticated geometric calculations for both physical borders (touching areas)
    and logical borders (non-touching areas), useful for map visualizations and directional indicators.
    """

    PROJECTION_POINT_IDENTIFIER = 'projection_point'
    PROJECTION_ANGLE_IDENTIFIER = 'projection_angle'
    BORDER_IS_PHYSICAL_IDENTIFIER = 'is_physical'
    BORDER_LINE_STRING_IDENTIFIER = 'geo_line_string'
    
    def __init__(self, area_model_gdf: gpd.GeoDataFrame):
        self.area_model_gdf = area_model_gdf
        self._validate_geometries()
    
    def _validate_geometries(self):
        """Ensure all geometries are valid."""
        self.area_model_gdf['geometry'] = self.area_model_gdf['geometry'].apply(
            lambda geom: geom if geom.is_valid else geom.buffer(0)
        )
    
    def calculate_border_geometry(
        self, 
        area_from: str, 
        area_to: str
    ) -> dict[str, Union[Point, float, LineString]]:
        """Calculate complete geometric properties for an area border.
        
        Args:
            area_from: Source area identifier
            area_to: Target area identifier
            
        Returns:
            Dictionary containing:
                - projection_point: Point for placing labels/arrows
                - projection_angle: Angle in degrees for directional indicators
                - border_line: LineString representing the border
                - is_physical: Whether areas share a common edge
        """
        midpoint, angle = self.get_area_border_midpoint_and_angle(area_from, area_to)
        
        if self.areas_touch(area_from, area_to):
            geom_from = self.get_area_geometry(area_from)
            geom_to = self.get_area_geometry(area_to)
            border_line = self._get_continuous_border_line(geom_from, geom_to)
            is_physical = True
        else:
            border_line = self.get_straight_line_between_areas(area_from, area_to)
            is_physical = False
        
        return {
            self.PROJECTION_POINT_IDENTIFIER: midpoint,
            self.PROJECTION_ANGLE_IDENTIFIER: angle,
            self.BORDER_LINE_STRING_IDENTIFIER: border_line,
            self.BORDER_IS_PHYSICAL_IDENTIFIER: is_physical
        }
    
    def areas_touch(self, area_from: str, area_to: str) -> bool:
        """Check if two areas share a common border."""
        geom_from = self.get_area_geometry(area_from)
        geom_to = self.get_area_geometry(area_to)
        return geom_from.touches(geom_to)
    
    def get_area_border_midpoint_and_angle(
        self, 
        area_from: str, 
        area_to: str
    ) -> tuple[Point, float]:
        """Get the midpoint and directional angle for a border.
        
        The angle points from area_from to area_to, suitable for directional indicators
        like arrows on maps.
            
        Returns:
            Tuple of (midpoint, angle_in_degrees)
        """
        geom_from = self.get_area_geometry(area_from)
        geom_to = self.get_area_geometry(area_to)
        
        if self.areas_touch(area_from, area_to):
            midpoint, angle = self._get_midpoint_and_angle_for_touching_areas(geom_from, geom_to)
        else:
            straight_line = self.get_straight_line_between_areas(area_from, area_to)
            midpoint, angle = self._get_midpoint_and_angle_from_line(straight_line)
        
        # Ensure angle points from area_from to area_to
        if not self._angle_points_to_target(geom_from, geom_to, midpoint, angle):
            angle = (angle + 180) % 360
        
        return midpoint, angle
    
    def get_area_geometry(self, area: str) -> Union[Polygon, MultiPolygon]:
        """Get the geometry for an area."""
        return self.area_model_gdf.loc[area].geometry.buffer(0)
    
    def get_straight_line_between_areas(self, area_from: str, area_to: str) -> LineString:
        """Get a straight line connecting two areas.
        
        For non-touching areas, finds the best line that avoids crossing other areas
        if possible.
        """
        if self.areas_touch(area_from, area_to):
            raise ValueError(f"Areas {area_from} and {area_to} touch - use border line instead")
        
        geom_from = self._get_largest_polygon(self.get_area_geometry(area_from))
        geom_to = self._get_largest_polygon(self.get_area_geometry(area_to))
        
        # Try centroid-to-centroid line first
        centroid_from = geom_from.centroid
        centroid_to = geom_to.centroid
        
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
        self,
        geom_from: Union[Polygon, MultiPolygon],
        geom_to: Union[Polygon, MultiPolygon]
    ) -> tuple[Point, float]:
        """Calculate midpoint and angle for areas that share a border."""
        border_line = self._get_continuous_border_line(geom_from, geom_to)
        midpoint = border_line.interpolate(0.5, normalized=True)
        
        # Get angle perpendicular to border
        start_to_end = self._get_straight_line_from_endpoints(border_line)
        border_bearing = self._calculate_bearing(start_to_end)
        perpendicular_angle = (border_bearing + 90) % 360
        
        return midpoint, perpendicular_angle
    
    def _get_midpoint_and_angle_from_line(self, line: LineString) -> tuple[Point, float]:
        """Get midpoint and bearing angle from a line."""
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
        """Check if angle points from source to target geometry."""
        centroid_from = geom_from.centroid
        centroid_to = geom_to.centroid
        
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
        """Get the shared border between two touching geometries."""
        if not geom_a.touches(geom_b):
            raise ValueError("Geometries do not touch")
        
        border = geom_a.intersection(geom_b)
        
        if isinstance(border, LineString):
            return border
        elif isinstance(border, MultiLineString):
            return self._merge_multilinestring(border)
        else:
            raise TypeError(f"Unexpected border type: {type(border)}")
    
    def _get_boundary_intersection(
        self,
        geom: Polygon,
        line: LineString,
        target_point: Point
    ) -> Point:
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
        """Check if a line crosses any areas except the excluded ones."""
        other_areas = self.area_model_gdf.drop(list(exclude_areas))
        return other_areas.geometry.crosses(line).any()
    
    def _find_non_crossing_line(
        self,
        area_from: str,
        area_to: str,
        min_clearance: float = 50000,  # meters in projected CRS
        num_points: int = 100
    ) -> Union[LineString, None]:
        """Find shortest line between areas that doesn't cross other areas.
        
        Args:
            area_from: Source area identifier
            area_to: Target area identifier
            min_clearance: Minimum distance from other areas (in projected CRS units)
            num_points: Number of points to test on each area boundary
            
        Returns:
            LineString if found, None otherwise
        """
        poly_from = self._get_largest_polygon(self.get_area_geometry(area_from))
        poly_to = self._get_largest_polygon(self.get_area_geometry(area_to))
        
        other_areas = self.area_model_gdf.drop([area_from, area_to])
        
        finder = NonCrossingPathFinder(
            poly_from,
            poly_to,
            other_areas,
            f"{area_from} to {area_to}"
        )
        
        return finder.find_shortest_path(min_clearance, num_points)
    
    def _get_largest_polygon(self, geom: Union[Polygon, MultiPolygon]) -> Polygon:
        """Extract the largest polygon from a MultiPolygon."""
        if isinstance(geom, Polygon):
            return geom
        return max(geom.geoms, key=lambda p: p.area)
    
    def _merge_multilinestring(self, mls: MultiLineString) -> LineString:
        """Merge a MultiLineString into a single continuous LineString."""
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
        """Create a straight line from start to end of a LineString."""
        return LineString([line.coords[0], line.coords[-1]])
    
    def _calculate_bearing(self, line: LineString) -> float:
        """Calculate compass bearing in degrees for a line.
        
        Args:
            line: LineString from which to calculate bearing
            
        Returns:
            Bearing in degrees (0-360, where 0 is north)
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
        """Calculate minimum angular difference between two angles."""
        diff = abs(angle1 - angle2) % 360
        return min(diff, 360 - diff)


class NonCrossingPathFinder:
    """Finds shortest path between polygons that doesn't cross other areas."""
    
    def __init__(
        self,
        polygon1: Polygon,
        polygon2: Polygon,
        other_areas: gpd.GeoDataFrame,
        name: str = None
    ):
        self.polygon1 = polygon1
        self.polygon2 = polygon2
        self.other_areas = other_areas
        self.name = name or "path"
    
    def find_shortest_path(
        self,
        min_clearance: float = 50000,
        num_points: int = 100,
        show_progress: bool = True
    ) -> Union[LineString, None]:
        """Find shortest path that maintains minimum clearance from other areas.
        
        Args:
            min_clearance: Minimum distance from other areas (in projected CRS units)
            num_points: Number of points to test on each polygon boundary
            show_progress: Whether to show progress bar
            
        Returns:
            Shortest valid LineString path, or None if no path found
        """
        # Buffer other areas for clearance check
        buffered_areas = self._buffer_areas(self.other_areas, min_clearance)
        
        # Get candidate points on boundaries
        points1 = self._get_boundary_points(self.polygon1, num_points)
        points2 = self._get_boundary_points(self.polygon2, num_points)
        
        # Generate all possible lines
        lines = [
            LineString([p1, p2])
            for p1, p2 in itertools.product(points1, points2)
        ]
        
        # Find shortest non-crossing line
        shortest_line = None
        min_length = float('inf')
        
        iterator = lines
        if show_progress:
            iterator = tqdm(lines, desc=f"Finding path for {self.name}")
        
        for line in iterator:
            if not buffered_areas.geometry.crosses(line).any():
                if line.length < min_length:
                    shortest_line = line
                    min_length = line.length
        
        return shortest_line
    
    def _buffer_areas(self, areas: gpd.GeoDataFrame, buffer_distance: float) -> gpd.GeoDataFrame:
        """Buffer areas by specified distance, handling CRS conversion."""
        areas_copy = areas.copy()
        
        # Convert to projected CRS for accurate buffering
        original_crs = areas_copy.crs
        if original_crs and original_crs.to_epsg() == 4326:
            areas_copy = areas_copy.to_crs(epsg=3857)
            areas_copy['geometry'] = areas_copy.buffer(buffer_distance)
            areas_copy = areas_copy.to_crs(original_crs)
        else:
            areas_copy['geometry'] = areas_copy.buffer(buffer_distance)
        
        return areas_copy
    
    def _get_boundary_points(self, polygon: Polygon, num_points: int) -> list[Point]:
        """Get evenly spaced points along polygon boundary."""
        boundary = polygon.boundary
        total_length = boundary.length
        
        points = []
        for i in range(num_points):
            distance = (i / num_points) * total_length
            point = boundary.interpolate(distance)
            points.append(point)
        
        return points


if __name__ == '__main__':
    # Example usage
    import geopandas as gpd
    
    # Load example data (assumes countries.geojson exists)
    gdf = gpd.read_file('data/countries.geojson').set_index('id')
    
    calculator = AreaBorderGeometryCalculator(gdf)
    
    # Test different border types
    test_pairs = [('NO', 'DE'), ('DE', 'PL'), ('PL', 'DE'), ('DE', 'GB')]
    
    for area_from, area_to in test_pairs:
        geometry_info = calculator.calculate_border_geometry(area_from, area_to)
        
        print(f"\n{area_from} → {area_to}:")
        print(f"  Projection point: {geometry_info['projection_point']}")
        print(f"  Angle: {geometry_info['projection_angle']:.1f}°")
        print(f"  Physical border: {geometry_info['is_physical']}")
        print(f"  Border length: {geometry_info['border_line'].length:.6f}")