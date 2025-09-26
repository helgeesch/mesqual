import pandas as pd
from shapely.geometry import Point, Polygon, MultiPolygon
import geopandas as gpd

from mesqual.energy_data_handling.area_accounting.model_generator_base import GeoModelGeneratorBase


class AreaModelGenerator(GeoModelGeneratorBase):
    """Generates comprehensive area model DataFrames from node-to-area mappings.
    
    This class creates detailed area model DataFrames that aggregate node-level data
    into area-level representations for energy system analysis. It supports
    automatic area discovery, node counting, and geographic representative point
    calculation for visualization and spatial analysis.
    
    The generator processes node model data with area assignments to create
    comprehensive area models suitable for energy system aggregation, market
    analysis, and spatial visualization workflows.
    
    Key Features:
        - Automatic area discovery from node-to-area mappings
        - Representative geographic point calculation for visualization
        - Integration with geometric area data (polygons, boundaries)
        - Support for different area granularities (countries, bidding zones, regions)
        - Robust handling of missing or incomplete area assignments
    
    MESQUAL Integration:
        Designed to work with MESQUAL's area accounting system, providing
        area model building capabilities that support spatial energy system analysis,
        capacity aggregation, and visualization workflows.
    
    Attributes:
        node_model_df (pd.DataFrame): Node-level data with area assignments
        area_column (str): Column name containing area identifiers
        geo_location_column (str): Column name containing geographic Point objects
    
    Examples:

        Basic area model generation:
        >>> import pandas as pd
        >>> from shapely.geometry import Point
        >>>
        >>> # Create node model with area assignments
        >>> node_data = pd.DataFrame({
        >>>     'voltage': [380, 380, 220, 380, 220],
        >>>     'country': ['DE', 'DE', 'FR', 'FR', 'BE'],
        >>>     'capacity_mw': [2000, 1500, 800, 1200, 600],
        >>>     'location': [Point(10, 52), Point(11, 53), Point(2, 48),
        >>>                  Point(3, 49), Point(4, 50)]
        >>> }, index=['DE1', 'DE2', 'FR1', 'FR2', 'BE1'])
        >>>
        >>> # Generate area model
        >>> generator = AreaModelGenerator(node_data, 'country')
        >>> area_model = generator.generate_area_model()
        >>> print(area_model)
                      node_count projection_point
            country
            DE               2    POINT (10.5 52.5)
            FR               2    POINT (2.5 48.5)
            BE               1    POINT (4 50)
        
        Enhanced area model with geometry:
        >>> import geopandas as gpd
        >>> from shapely.geometry import Polygon
        >>>
        >>> # Create area geometries
        >>> area_polygons = gpd.GeoDataFrame({
        >>>     'geometry': [
        >>>         Polygon([(9, 51), (12, 51), (12, 54), (9, 54)]),  # DE
        >>>         Polygon([(1, 47), (4, 47), (4, 50), (1, 50)]),   # FR
        >>>         Polygon([(3, 49), (5, 49), (5, 51), (3, 51)])    # BE
        >>>     ]
        >>> }, index=['DE', 'FR', 'BE'])
        >>>
        >>> # Enhance with geometry
        >>> area_model_geo = generator.enhance_with_geometry(area_model, area_polygons)
        >>> print(f"Enhanced model has geometry: {'geometry' in area_model_geo.columns}")
        
        Custom enhancement workflow:
        >>> # Step-by-step area model building
        >>> base_model = generator.generate_base_area_model_from_area_names_in_node_model_df()
        >>> enhanced_model = generator.enhance_area_model_df_by_adding_node_count_per_area(base_model)
        >>> final_model = generator.enhance_area_model_df_by_adding_representative_geo_point(enhanced_model)
        >>> print(f"Created area model with {len(final_model)} areas")
    
    Energy Domain Context:
        - Area models are fundamental for energy system analysis, enabling:
            - Projection of node-level data to area-level data (e.g. nodal prices -> area prices)
            - Market zone aggregation and analysis
            - Regional energy balance studies
            - ...
    """
    
    def __init__(
            self,
            node_model_df: pd.DataFrame,
            area_column: str,
            geo_location_column: str = None,
    ):
        """Initialize the area model generator.
        
        Args:
            node_model_df: DataFrame containing node-level data with area assignments.
                Must contain area_column with area identifiers for each node.
                May contain geographic Point objects for spatial analysis.
            area_column: Column name in node_model_df containing area assignments
                (e.g., 'country', 'bidding_zone', 'market_region', 'control_area').
            geo_location_column: Column name containing geographic Point objects
                for representative point calculation. If None, automatically
                detects column containing Point geometries.
                
        Raises:
            ValueError: If area_column is not found in node_model_df columns.
            
        Example:

            >>> import pandas as pd
            >>> from shapely.geometry import Point
            >>> 
            >>> # Node data with area assignments
            >>> nodes = pd.DataFrame({
            >>>     'voltage_kv': [380, 220, 380, 150],
            >>>     'country': ['DE', 'DE', 'FR', 'FR'], 
            >>>     'bidding_zone': ['DE_LU', 'DE_LU', 'FR', 'FR'],
            >>>     'coordinates': [Point(10, 52), Point(11, 53), Point(2, 48), Point(3, 49)]
            >>> }, index=['DE1', 'DE2', 'FR1', 'FR2'])
            >>> 
            >>> # Initialize for country-level analysis
            >>> generator = AreaModelGenerator(nodes, 'country', 'coordinates')
            >>> 
            >>> # Or let it auto-detect geographic column
            >>> generator = AreaModelGenerator(nodes, 'bidding_zone')
        """
        self.node_model_df = node_model_df
        self.area_column = area_column
        self.geo_location_column = geo_location_column or self._identify_geo_location_column()
        self._validate_inputs()
    
    def _validate_inputs(self):
        if self.area_column not in self.node_model_df.columns:
            raise ValueError(
                f"Area column '{self.area_column}' not found in node_model_df. "
                f"Available columns: {list(self.node_model_df.columns)}"
            )

    def _identify_geo_location_column(self) -> str | None:
        for c in self.node_model_df.columns:
            if all(isinstance(i, Point) or i is None for i in self.node_model_df[c].values):
                return c
        return None

    def generate_base_area_model_from_area_names_in_node_model_df(self) -> pd.DataFrame:
        """Generate base area model DataFrame from unique area names in node data.
        
        Creates a minimal area model DataFrame containing only the unique area
        identifiers found in the node model data. This forms the foundation
        for building comprehensive area models.
        
        Returns:
            pd.DataFrame: Base area model with area identifiers as index.
                Contains no additional columns - serves as starting point
                for enhancement with node counts, geographic data, etc.
                
        Example:

            >>> generator = AreaModelGenerator(node_data, 'country')
            >>> base_model = generator.generate_base_area_model_from_area_names_in_node_model_df()
            >>> print(base_model)
                Empty DataFrame
                Columns: []
                Index: ['DE', 'FR', 'BE']
            
        Note:
            Areas with None or NaN values in the area_column are excluded
            from the generated model.
        """
        unique_areas = self.node_model_df[self.area_column].dropna().unique()
        area_model_df = pd.DataFrame(index=unique_areas)
        area_model_df.index.name = self.area_column
        return area_model_df

    def ensure_completeness_of_area_model_df(self, area_model_df: pd.DataFrame) -> pd.DataFrame:
        """Ensure area model contains all areas present in node data.
        
        Validates and extends an existing area model DataFrame to include
        any areas found in the node data that might be missing from the
        provided area model. This is useful when working with predefined
        area models that may not cover all areas in the dataset.
        
        Args:
            area_model_df: Existing area model DataFrame to validate and extend.
            
        Returns:
            pd.DataFrame: Complete area model containing all areas from node data.
                Existing data is preserved, new areas are added with NaN values
                for existing columns.
                
        Example:

            >>> # Predefined area model missing some areas
            >>> partial_model = pd.DataFrame({
            >>>     'max_price': [5000, 3000]
            >>> }, index=['DE', 'FR'])
            >>> 
            >>> # Ensure completeness (adds 'BE' if present in node data)
            >>> complete_model = generator.ensure_completeness_of_area_model_df(partial_model)
            >>> print(complete_model)
                          max_price
                country
                DE             5000
                FR             3000
                BE              NaN
            
        Use Case:
            Essential for maintaining data consistency when combining
            predefined area models with dynamic node-based area discovery.
        """
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
        """Enhance area model by adding node count statistics per area.
        
        Aggregates the number of nodes assigned to each area and adds this
        information to the area model DataFrame. Node counts are essential
        for understanding infrastructure density and capacity distribution.
        
        Args:
            area_model_df: Base area model DataFrame to enhance.
            node_count_column_name: Name for the new node count column.
                Defaults to 'node_count'.
                
        Returns:
            pd.DataFrame: Enhanced area model with node count column added.
                Existing data is preserved, node counts are added for all areas.
                Areas not present in node data will have NaN node counts.
                
        Example:

            >>> base_model = generator.generate_base_area_model_from_area_names_in_node_model_df()
            >>> enhanced_model = generator.enhance_area_model_df_by_adding_node_count_per_area(base_model)
            >>> print(enhanced_model)
                          node_count
                country
                DE               2
                FR               2
                BE               1
            
            >>> # Custom column name
            >>> enhanced_model = generator.enhance_area_model_df_by_adding_node_count_per_area(
            >>>     base_model, 'infrastructure_count'
            >>> )
        """
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
        """Enhance area model by adding representative geographic points for
        labeling and KPI printing in map visualizations.
        
        Calculates representative geographic points for each area based on
        either area geometries (if available) or node locations within each area.
        These points are useful for visualization, labeling, and spatial analysis.
        
        The method supports two calculation modes:
        1. Geometry-based: Uses area polygon centroids or representative points
        2. Node-based: Calculates centroid from node locations within each area
        
        Args:
            area_model_df: Area model DataFrame to enhance. Can be regular DataFrame
                or GeoDataFrame with 'geometry' column.
            target_column_name: Name for the new representative point column.
                Defaults to 'projection_point'.
            round_point_decimals: Number of decimal places for coordinate rounding.
                Set to None to disable rounding. Defaults to 4.
                
        Returns:
            pd.DataFrame: Enhanced area model with representative points added.
                Points are added as Shapely Point objects suitable for mapping
                and spatial analysis.
                
        Raises:
            TypeError: If geo_location_column contains non-Point objects.
            
        Example:

            >>> # Node-based representative points
            >>> enhanced_model = generator.enhance_area_model_df_by_adding_representative_geo_point(base_model)
            >>> print(enhanced_model)
                          projection_point
                country
                DE        POINT (10.5 52.5)
                FR        POINT (2.5 48.5)
                BE        POINT (4 50)
            
            >>> # With custom column name and precision
            >>> enhanced_model = generator.enhance_area_model_df_by_adding_representative_geo_point(
            >>>     base_model, 'center_point', round_point_decimals=2
            >>> )
            
            >>> # Access coordinates for mapping
            >>> center = enhanced_model.loc['DE', 'projection_point']
            >>> print(f"DE center: {center.x:.2f}, {center.y:.2f}")
        """
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
                    enhanced_df.loc[area, target_column_name] = round_point(self.get_representative_area_point(geo))
            elif self.geo_location_column:
                nodes = self.node_model_df.loc[self.node_model_df[self.area_column] == area, self.geo_location_column]
                nodes = nodes.dropna()
                if not nodes.empty:
                    locations = [n for n in nodes.values if n is not None]
                    if not all(isinstance(i, Point) for i in locations):
                        raise TypeError(
                            f'Geographic location column "{self.geo_location_column}" must contain only '
                            f'Point objects. Found: {[type(i).__name__ for i in locations if not isinstance(i, Point)]}'
                        )
                    representative_point = self._compute_representative_point_from_cloud_of_2d_points(locations)
                    enhanced_df.loc[area, target_column_name] = round_point(representative_point)
        return enhanced_df

    def generate_area_model(self) -> pd.DataFrame:
        """Generate complete area model with node counts and representative points.
        
        Creates a comprehensive area model DataFrame by combining base area
        discovery, node count aggregation, and representative geographic point
        calculation. This is the main method for generating complete area models.
        
        The generated model includes:
            - All unique areas from node data
            - Node count per area for capacity/infrastructure analysis
            - Representative geographic points for visualization
        
        Returns:
            pd.DataFrame: Complete area model with node counts and geographic data.
                Index contains area identifiers, columns include 'node_count'
                and 'projection_point' (if geographic data available).
                
        Example:

            >>> generator = AreaModelGenerator(node_data, 'country')
            >>> area_model = generator.generate_area_model()
            >>> print(area_model)
                          node_count projection_point
                country
                DE               2    POINT (10.5 52.5)
                FR               2    POINT (2.5 48.5)
                BE               1    POINT (4 50)
        """
        area_model_df = self.generate_base_area_model_from_area_names_in_node_model_df()
        area_model_df = self.enhance_area_model_df_by_adding_node_count_per_area(area_model_df)
        area_model_df = self.enhance_area_model_df_by_adding_representative_geo_point(area_model_df)
        return area_model_df
    
    def enhance_with_geometry(
        self,
        area_model_df: pd.DataFrame,
        area_gdf: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
        """Enhance area model with geometric polygon data for spatial analysis.
        
        Integrates area polygon geometries from a GeoDataFrame into the area model,
        enabling advanced spatial analysis, visualization, and border calculations.
        The method matches areas by index and creates a proper GeoDataFrame output.
        
        Args:
            area_model_df: Area model DataFrame to enhance with geometry.
            area_gdf: GeoDataFrame containing area polygon geometries.
                Must have 'geometry' column with Polygon or MultiPolygon objects.
                Areas are matched by index values.
                
        Returns:
            gpd.GeoDataFrame: Enhanced area model as GeoDataFrame with geometry column.
                All original data is preserved, geometry column is added for areas
                that exist in both DataFrames. Missing geometries are set to None.
                
        Example:

            >>> import geopandas as gpd
            >>> from shapely.geometry import Polygon
            >>> 
            >>> # Create area geometries
            >>> area_polygons = gpd.GeoDataFrame({
            >>>     'area_name': ['Germany', 'France', 'Belgium'],
            >>>     'geometry': [
            >>>         Polygon([(9, 51), (12, 51), (12, 54), (9, 54)]),
            >>>         Polygon([(1, 47), (4, 47), (4, 50), (1, 50)]),
            >>>         Polygon([(3, 49), (5, 49), (5, 51), (3, 51)])
            >>>     ]
            >>> }, index=['DE', 'FR', 'BE'])
            >>> 
            >>> # Enhance area model with geometry
            >>> geo_model = generator.enhance_with_geometry(area_model, area_polygons)
            >>> print(f"Model has geometry: {isinstance(geo_model, gpd.GeoDataFrame)}")
            >>> 
            >>> # Use for spatial operations
            >>> total_area = geo_model['geometry'].area.sum()
            >>> print(f"Total area: {total_area:.0f} square units")
        """
        enhanced_df = area_model_df.copy()
        if 'geometry' not in enhanced_df.columns:
            enhanced_df['geometry'] = None
        for area in area_model_df.index:
            if area in area_gdf.index:
                enhanced_df.loc[area, 'geometry'] = area_gdf.loc[area, 'geometry']
        if not isinstance(enhanced_df, gpd.GeoDataFrame):
            enhanced_df = gpd.GeoDataFrame(enhanced_df, geometry='geometry')
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
