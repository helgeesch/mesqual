import pandas as pd
from shapely.geometry import Point, Polygon, MultiPolygon
import geopandas as gpd

from mesqual.energy_data_handling.area_accounting.area_membership import AreaMembership
from mesqual.energy_data_handling.area_accounting.model_generator_base import GeoModelGeneratorBase


class AreaModelGenerator(GeoModelGeneratorBase, AreaMembership):
    """Area model from the area of each node: one row per area.

    The model has the node count and a representative point per area.
    `generate_area_model` takes the point from the node locations.
    `enhance_with_geometry` adds only the area geometries and keeps the
    existing points.

    Attributes:
        node_model_df (pd.DataFrame): Node-level data with area assignments
        area_column (str): Column name containing area identifiers
        geo_location_column (str): Column name containing geographic Point objects

    Examples:
        >>> generator = AreaModelGenerator(dataset.fetch('Node.Model'), 'country')
        >>> generator.generate_area_model()
                 node_count         projection_point
        country
        C1              2.0   POINT (6.3888 47.0112)
        C2              2.0  POINT (10.3888 47.0112)
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
        """
        self.node_model_df = node_model_df
        self.area_column = area_column
        self.geo_location_column = geo_location_column or self._identify_geo_location_column()
        self._check_area_column()

    def _identify_geo_location_column(self) -> str | None:
        for c in self.node_model_df.columns:
            if all(isinstance(i, Point) or i is None for i in self.node_model_df[c].values):
                return c
        return None

    def generate_base_area_model_from_area_names_in_node_model_df(self) -> pd.DataFrame:
        """Generate base area model DataFrame from unique area names in node data.

        Creates a minimal area model DataFrame containing only the unique area
        identifiers found in the node model data. This forms the foundation
        for building area models.

        Returns:
            pd.DataFrame: Base area model with area identifiers as index.
                Contains no additional columns - serves as starting point
                for enhancement with node counts, geographic data, etc.

        Note:
            Nodes without an area (`AreaMembership.has_area`) add no area
            to the generated model.
        """
        unique_areas = self.get_node_areas().unique()
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

        Examples:
            >>> generator = AreaModelGenerator(dataset.fetch('Node.Model'), 'country')
            >>> partial = pd.DataFrame({'max_price': [3000.0]}, index=pd.Index(['C1'], name='country'))
            >>> generator.ensure_completeness_of_area_model_df(partial)
                     max_price
            country
            C1          3000.0
            C2             NaN
        """
        complete_set = area_model_df.index.to_list()
        for a in self.get_node_areas().unique():
            if a not in complete_set:
                complete_set.append(a)
        return area_model_df.reindex(complete_set)

    def enhance_area_model_df_by_adding_node_count_per_area(
        self, area_model_df: pd.DataFrame, node_count_column_name: str = 'node_count'
    ) -> pd.DataFrame:
        """Enhance area model by adding node count statistics per area.

        Aggregates the number of nodes assigned to each area and adds this
        information to the area model DataFrame.

        Args:
            area_model_df: Base area model DataFrame to enhance.
            node_count_column_name: Name for the new node count column.
                Defaults to 'node_count'.

        Returns:
            pd.DataFrame: Enhanced area model with node count column added.
                Existing data is preserved, node counts are added for all areas.
                Areas not present in node data will have NaN node counts.
        """
        enhanced_df = area_model_df.copy()
        node_counts = self.get_node_areas().value_counts().to_dict()
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

        An area that already has a point keeps it. If the model has a
        `geometry` column, the point comes from the polygon of the area; an
        area without a polygon gets no point. Else the point comes from the
        locations of the nodes of the area (`geo_location_column`).

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

        One row per area of the node model, with the node count and the
        representative point of the node locations.

        Returns:
            pd.DataFrame: Complete area model. Index contains area identifiers,
                columns are 'node_count' and 'projection_point'.
                'projection_point' is None where no node location is available.
        """
        area_model_df = self.generate_base_area_model_from_area_names_in_node_model_df()
        area_model_df = self.enhance_area_model_df_by_adding_node_count_per_area(area_model_df)
        area_model_df = self.enhance_area_model_df_by_adding_representative_geo_point(area_model_df)
        return area_model_df

    def enhance_with_geometry(self, area_model_df: pd.DataFrame, area_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Add the area polygons of a GeoDataFrame to the area model.

        Areas are matched by index. The result is a GeoDataFrame.

        Args:
            area_model_df: Area model DataFrame to enhance with geometry.
            area_gdf: GeoDataFrame containing area polygon geometries.
                Must have 'geometry' column with Polygon or MultiPolygon objects.
                Areas are matched by index values.

        Returns:
            gpd.GeoDataFrame: Enhanced area model as GeoDataFrame with geometry column.
                All original data is preserved, geometry column is added for areas
                that exist in both DataFrames. Missing geometries are set to None.

        Examples:
            >>> generator = AreaModelGenerator(dataset.fetch('Node.Model'), 'country')
            >>> areas = generator.enhance_with_geometry(generator.generate_area_model(), dataset.fetch('Country.Model'))
            >>> areas['geometry'].tolist()
            [<POLYGON ((9 46, 9 50, 5 50, 5 46, 9 46))>, <POLYGON ((13 46, 13 50, 9 50, 9 46, 13 46))>]
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
