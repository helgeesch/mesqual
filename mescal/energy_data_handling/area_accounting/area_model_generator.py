from typing import Union, List
import pandas as pd
from shapely.geometry import Point, Polygon, MultiPolygon
import geopandas as gpd


class AreaModelGenerator:
    """Generates area model DataFrame from node-to-area mapping."""
    
    def __init__(
            self,
            node_model_df: pd.DataFrame,
            area_column: str,
            geo_location_column: str = None,
    ):
        self.node_model_df = node_model_df
        self.area_column = area_column
        self.geo_location_column = geo_location_column or self._identify_geo_location_column()
        self._validate_inputs()
    
    def _validate_inputs(self):
        if self.area_column not in self.node_model_df.columns:
            raise ValueError(f"Column '{self.area_column}' not found in node_model_df")

    def _identify_geo_location_column(self) -> str | None:
        for c in self.node_model_df.columns:
            if all(isinstance(i, Point) or i is None for i in self.node_model_df[c].values):
                return c
        return None

    def generate_base_area_model_from_area_names_in_node_model_df(self) -> pd.DataFrame:
        unique_areas = self.node_model_df[self.area_column].dropna().unique()
        area_model_df = pd.DataFrame(index=unique_areas)
        area_model_df.index.name = self.area_column
        return area_model_df

    def ensure_completeness_of_area_model_df(self, area_model_df: pd.DataFrame) -> pd.DataFrame:
        complete_set = area_model_df.index.to_list()
        for a in self.node_model_df[self.area_column].unique():
            if a is not None and a not in complete_set:
                complete_set.append(a)
        return area_model_df.reindex(complete_set)

    def enhance_area_model_df_by_adding_node_count_per_area(
            self,
            area_model_df: pd.DataFrame,
            node_count_column_name: str = 'node_count'
    ) -> pd.DataFrame:
        enhanced_df = area_model_df.copy()
        node_counts = self.node_model_df[self.area_column].value_counts().to_dict()
        for node, count in node_counts.items():
            if node in enhanced_df.index:
                enhanced_df.loc[node, node_count_column_name] = count
        return enhanced_df

    def enhance_area_model_df_by_adding_representative_geo_point(
            self,
            area_model_df: pd.DataFrame | gpd.GeoDataFrame,
            target_column_name: str = 'projection_point',
            round_point_decimals: int = 4,
    ) -> pd.DataFrame:
        enhanced_df = area_model_df.copy()

        def round_point(point: Point | None) -> Point | None:
            if point is None or round_point_decimals is None:
                return point
            return type(point)(round(point.x, round_point_decimals), round(point.y, round_point_decimals))

        if target_column_name not in enhanced_df:
            enhanced_df[target_column_name] = None

        for area in enhanced_df.index:
            if pd.notna(enhanced_df.loc[area, target_column_name]):
                continue
            if 'geometry' in enhanced_df.columns:
                geo = enhanced_df.loc[area, 'geometry']
                if pd.notna(geo) and isinstance(geo, (Polygon, MultiPolygon)):
                    enhanced_df.loc[area, target_column_name] = round_point(geo.representative_point())
            if self.geo_location_column:
                nodes = node_model_df.loc[node_model_df[self.area_column] == area, self.geo_location_column]
                nodes = nodes.dropna()
                if not nodes.empty:
                    locations = [n for n in nodes.values if n is not None]
                    if not all(isinstance(i, Point) for i in locations):
                        raise TypeError(f'Can only handle Point objects in node_point_column {self.geo_location_column}.')
                    representative_point = self._compute_representative_point_from_cloud_of_2d_points(locations)
                    enhanced_df.loc[area, target_column_name] = round_point(representative_point)
        return enhanced_df

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

    def generate_area_model(self) -> pd.DataFrame:
        area_model_df = self.generate_base_area_model_from_area_names_in_node_model_df()
        area_model_df = self.enhance_area_model_df_by_adding_node_count_per_area(area_model_df)
        area_model_df = self.enhance_area_model_df_by_adding_representative_geo_point(area_model_df)
        return area_model_df
    
    def enhance_with_geometry(
        self,
        area_model_df: pd.DataFrame,
        area_gdf: gpd.GeoDataFrame
    ) -> pd.DataFrame:
        """Enhance area model with geometries from a GeoDataFrame."""
        enhanced_df = area_model_df.copy()
        for area in area_model_df.index:
            if area in area_gdf.index:
                enhanced_df.loc[area, 'geometry'] = area_gdf.loc[area, 'geometry']
        return enhanced_df


if __name__ == '__main__':
    # Create dummy node model data
    node_model_df = pd.DataFrame({
        'voltage': [380, 380, 220, 380, 220],
        'country': ['DE', 'DE', 'FR', 'FR', 'BE'],
        'bidding_zone': ['DE_LU', 'DE_LU', 'FR', 'FR', 'BE'],
        'location': [Point(10, 52), Point(11, 53), Point(2, 48), Point(3, 49), Point(4, 50)]
    }, index=['DE1', 'DE2', 'FR1', 'FR2', 'BE1'])
    
    # Generate area model for countries
    generator = AreaModelGenerator(node_model_df, 'country')
    area_model = generator.generate_area_model()
    print("Area Model (Countries):")
    print(area_model)
    print()
    
    # Generate area model for bidding zones
    generator_bz = AreaModelGenerator(node_model_df, 'bidding_zone')
    area_model_bz = generator_bz.generate_area_model()
    print("Area Model (Bidding Zones):")
    print(area_model_bz)
