"""Border model generation for energy system area connectivity analysis.

This module provides functionality for identifying and modeling borders between
energy system areas based on line topologies. It supports the
creation of comprehensive border_model_dfs that capture directional relationships,
naming conventions, and geometric properties essential for energy systems analysis.

Key Capabilities:
    - Automatic border identification from line topology
    - Standardized border naming conventions with directional awareness
    - Integration with geometric border calculators
    - Network graph generation for area connectivity analysis
    - Support for both physical and logical borders (geographically touching borders vs geographically separated borders)

Typical Energy Use Cases:
    - Modeling interconnections between countries, control areas, or market zones
    - Cross-border capacity and flow analysis
    - Network visualization and analysis

MESQUAL Integration:
    This module integrates with MESQUAL's area accounting system to provide
    border_model_df building capabilities that support spatial energy system analysis
    and cross-border flow calculations.
"""
from __future__ import annotations

from typing import Tuple
import pandas as pd
import networkx as nx

from mesqual.energy_data_handling.area_accounting.border_model_geometry_calculator import AreaBorderGeometryCalculator


class AreaBorderNamingConventions:
    """Standardized naming conventions for energy system area borders.
    
    This class provides consistent naming patterns for borders between energy
    system areas (countries, bidding zones, market regions). It ensures
    standardized naming across different analysis workflows and supports
    bidirectional relationship management.
    
    The naming system supports:
        - Configurable separators and prefixes/suffixes
        - Bidirectional border identification (A-B and B-A)
        - Alphabetically sorted canonical border names
        - Consistent column naming for source and target areas
    
    Key Features:
        - Configurable naming patterns for different use cases
        - Automatic opposite border name generation
        - Alphabetical sorting for canonical border representation
        - Consistent identifier generation for database/DataFrame columns
    
    Attributes:
        JOIN_AREA_NAMES_BY (str): Separator for area names in border identifiers
        SOURCE_AREA_IDENTIFIER_SUFFIX (str): Suffix for source area column names
        TARGET_AREA_IDENTIFIER_SUFFIX (str): Suffix for target area column names
        OPPOSITE_BORDER_IDENTIFIER (str): Column name for opposite border references
        SORTED_BORDER_IDENTIFIER (str): Column name for alphabetically sorted borders
        NAME_IS_ALPHABETICALLY_SORTED_IDENTIFIER (str): Boolean indicator column
    
    Example:

        >>> conventions = AreaBorderNamingConventions('country')
        >>> border_name = conventions.get_area_border_name('DE', 'FR')
        >>> print(border_name)  # 'DE - FR'
        >>> opposite = conventions.get_opposite_area_border_name(border_name)
        >>> print(opposite)  # 'FR - DE'
    """

    JOIN_AREA_NAMES_BY = ' - '
    SOURCE_AREA_IDENTIFIER_PREFIX = ''
    TARGET_AREA_IDENTIFIER_PREFIX = ''
    SOURCE_AREA_IDENTIFIER_SUFFIX = '_from'
    TARGET_AREA_IDENTIFIER_SUFFIX = '_to'
    OPPOSITE_BORDER_IDENTIFIER = 'opposite_border'
    SORTED_BORDER_IDENTIFIER = 'sorted_border'
    NAME_IS_ALPHABETICALLY_SORTED_IDENTIFIER = 'name_is_alphabetically_sorted'
    PROJECTION_POINT_IDENTIFIER = 'projection_point'
    AZIMUTH_ANGLE_IDENTIFIER = 'azimuth_angle'
    BORDER_IS_PHYSICAL_IDENTIFIER = 'is_physical'
    BORDER_LINE_STRING_IDENTIFIER = 'geo_line_string'

    def __init__(
            self,
            area_column: str,
            border_identifier: str = None,
            source_area_identifier: str = None,
            target_area_identifier: str = None,
    ):
        """Initialize border naming conventions.
        
        Args:
            area_column: Name of the area column (e.g., 'country', 'bidding_zone')
            border_identifier: Custom name for border identifier column.
                Defaults to '{area_column}_border'
            source_area_identifier: Custom name for source area column.
                Defaults to '{area_column}_from'
            target_area_identifier: Custom name for target area column.
                Defaults to '{area_column}_to'
                
        Example:

            >>> # Standard naming
            >>> conventions = AreaBorderNamingConventions('country')
            >>> print(conventions.border_identifier)  # 'country_border'
            >>> 
            >>> # Custom naming
            >>> conventions = AreaBorderNamingConventions(
            ...     'bidding_zone',
            ...     border_identifier='interconnection',
            ...     source_area_identifier='origin_zone'
            ... )
        """
        self.area_column = area_column
        self.border_identifier = border_identifier or self._default_border_identifier()
        self.source_area_identifier = source_area_identifier or self._default_source_area_identifier()
        self.target_area_identifier = target_area_identifier or self._default_target_area_identifier()

    def _default_border_identifier(self) -> str:
        return f'{self.area_column}_border'

    def _default_source_area_identifier(self) -> str:
        return f'{self.SOURCE_AREA_IDENTIFIER_PREFIX}{self.area_column}{self.SOURCE_AREA_IDENTIFIER_SUFFIX}'

    def _default_target_area_identifier(self) -> str:
        return f'{self.TARGET_AREA_IDENTIFIER_PREFIX}{self.area_column}{self.TARGET_AREA_IDENTIFIER_SUFFIX}'

    def get_area_border_name(self, area_from: str, area_to: str) -> str:
        """Generate standardized border name from source and target areas.
        
        Args:
            area_from: Source area identifier (e.g., 'DE', 'FR_North')
            area_to: Target area identifier (e.g., 'FR', 'DE_South')
            
        Returns:
            str: Formatted border name using the configured separator (e.g. 'DE - FR', 'FR_North - DE_South')
            
        Example:

            >>> conventions = AreaBorderNamingConventions('country')
            >>> border_name = conventions.get_area_border_name('DE', 'FR')
            >>> print(border_name)  # 'DE - FR'
        """
        return f'{area_from}{self.JOIN_AREA_NAMES_BY}{area_to}'

    def decompose_area_border_name_to_areas(self, border_name: str) -> Tuple[str, str]:
        """Extract source and target area names from border identifier.
        
        Args:
            border_name: Border name in standard format (e.g., 'DE - FR')
            
        Returns:
            Tuple[str, str]: Source and target area names
            
        Raises:
            ValueError: If border_name doesn't contain the expected separator
            
        Example:

            >>> conventions = AreaBorderNamingConventions('country')
            >>> area_from, area_to = conventions.decompose_area_border_name_to_areas('DE - FR')
            >>> print(f"From: {area_from}, To: {area_to}")  # From: DE, To: FR
        """
        area_from, area_to = border_name.split(self.JOIN_AREA_NAMES_BY)
        return area_from, area_to

    def get_opposite_area_border_name(self, border_name: str) -> str:
        """Generate the opposite direction border name.
        
        Args:
            border_name: Original border name (e.g., 'DE - FR')
            
        Returns:
            str: Opposite direction border name (e.g., 'FR - DE')
            
        Example:

            >>> conventions = AreaBorderNamingConventions('country')
            >>> opposite = conventions.get_opposite_area_border_name('DE - FR')
            >>> print(opposite)  # 'FR - DE'
            
        Energy Domain Context:
            Energy flows and capacities are often directional, requiring
            tracking of both A→B and B→A relationships for comprehensive
            border analysis.
        """
        area_from, area_to = self.decompose_area_border_name_to_areas(border_name)
        return self.get_area_border_name(area_to, area_from)

    def get_alphabetically_sorted_border(self, border_name: str) -> str:
        """Generate alphabetically sorted canonical border name.
        
        Creates a canonical representation where area names are sorted
        alphabetically, useful for identifying unique borders regardless
        of direction specification, or for matching borders of opposite direction.
        
        Args:
            border_name: Border name in any direction (e.g., 'FR - DE' or 'DE - FR')
            
        Returns:
            str: Alphabetically sorted border name (e.g., 'DE - FR')
            
        Example:

            >>> conventions = AreaBorderNamingConventions('country')
            >>> sorted_border = conventions.get_alphabetically_sorted_border('FR - DE')
            >>> print(sorted_border)  # 'DE - FR'
            
        Use Case:
            Canonical naming is essential for border deduplication and
            consistent reference in energy system databases and analysis.
        """
        area_from, area_to = self.decompose_area_border_name_to_areas(border_name)
        return self.get_area_border_name(*list(sorted([area_from, area_to])))


class AreaBorderModelGenerator(AreaBorderNamingConventions):
    """Generates comprehensive border models from energy system topology.
    
    This class analyzes line connectivity and node-to-area mappings
    to automatically identify borders between energy system areas. It creates
    a comprehensive border_model_df with standardized naming, directional relationships,
    and integration points for geometric analysis.
    
    The generator processes line topology data to identify cross-area connections.
    It supports bidirectional relationship tracking and provides network graph
    representations for connectivity analysis.
    
    Key Features:
        - Automatic border discovery from line topology
        - Bidirectional border relationship management
        - Standardized naming conventions with configurable patterns
        - Network graph generation for connectivity analysis
        - Integration with geometric border calculators
        - Support for different area granularities (countries, bidding zones, etc.)
    
    MESQUAL Integration:
        Designed to work with MESQUAL's area accounting system, providing
        border modeling capabilities that integrate with flow calculators,
        capacity analyzers, and visualization tools.
    
    Attributes:
        line_model_df (pd.DataFrame): Transmission line data with topology information
        node_model_df (pd.DataFrame): Node data with area assignments
        node_from_col (str): Column name for line source nodes
        node_to_col (str): Column name for line target nodes
        node_to_area_map (dict): Mapping from nodes to their assigned areas
    
    Example:

        >>> # Create border model from transmission data
        >>> generator = AreaBorderModelGenerator(
        ...     node_df, line_df, 'country', 'node_from', 'node_to'
        ... )
        >>> border_model = generator.generate_area_border_model()
        >>> print(f"Found {len(border_model)} directional borders")
    """

    def __init__(
        self, 
        node_model_df: pd.DataFrame,
        line_model_df: pd.DataFrame,
        area_column: str,
        node_from_col: str,
        node_to_col: str,
        border_identifier: str = None,
        source_area_identifier: str = None,
        target_area_identifier: str = None,
    ):
        """Initialize the area border model generator.
        
        Args:
            node_model_df: DataFrame containing node-level data with area assignments.
                Must contain area_column with area identifiers for each node.
            line_model_df: DataFrame containing transmission line topology data.
                Must contain node_from_col and node_to_col with node identifiers.
            area_column: Column name in node_model_df containing area assignments
                (e.g., 'country', 'bidding_zone', 'market_region')
            node_from_col: Column name in line_model_df for source node identifiers
            node_to_col: Column name in line_model_df for target node identifiers
            border_identifier: Custom border column name (optional)
            source_area_identifier: Custom source area column name (optional)
            target_area_identifier: Custom target area column name (optional)
            
        Raises:
            ValueError: If required columns are not found in input DataFrames
            
        Example:

            >>> generator = AreaBorderModelGenerator(
            ...     nodes_df=node_data,
            ...     lines_df=transmission_data,
            ...     area_column='bidding_zone',
            ...     node_from_col='bus_from',
            ...     node_to_col='bus_to'
            ... )
        """
        super().__init__(area_column, border_identifier, source_area_identifier, target_area_identifier)
        self.line_model_df = line_model_df
        self.node_model_df = node_model_df
        self.node_from_col = node_from_col
        self.node_to_col = node_to_col

        self._validate_inputs()
        self.node_to_area_map = self._create_node_to_area_map()

    def _validate_inputs(self):
        if self.area_column not in self.node_model_df.columns:
            raise ValueError(
                f"Area column '{self.area_column}' not found in node_model_df. "
                f"Available columns: {list(self.node_model_df.columns)}"
            )
        if self.node_from_col not in self.line_model_df.columns:
            raise ValueError(
                f"Source node column '{self.node_from_col}' not found in line_model_df. "
                f"Available columns: {list(self.line_model_df.columns)}"
            )
        if self.node_to_col not in self.line_model_df.columns:
            raise ValueError(
                f"Target node column '{self.node_to_col}' not found in line_model_df. "
                f"Available columns: {list(self.line_model_df.columns)}"
            )

    def _create_node_to_area_map(self) -> dict:
        """Create mapping from node identifiers to their assigned areas.
        
        Returns:
            dict: Mapping from node IDs to area assignments
            
        Note:
            Nodes with None or NaN area assignments are included in the mapping
            but will be filtered out during border identification.
        """
        return self.node_model_df[self.area_column].to_dict()
    
    def generate_area_border_model(self) -> pd.DataFrame:
        """Generate comprehensive border model with all relationship data.
        
        Analyzes transmission line topology to identify borders between areas,
        creating a comprehensive DataFrame with directional relationships,
        naming conventions, and reference data for further analysis.
        
        The generated model includes:
            - Border identifiers in both directions (A→B and B→A)
            - Source and target area columns
            - Opposite border references for bidirectional analysis
            - Alphabetically sorted canonical border names
            - Boolean indicators for alphabetical sorting
        
        Returns:
            pd.DataFrame: Comprehensive border model indexed by border identifiers.
                Returns empty DataFrame with proper column structure if no borders found.
                
        Example:

            >>> border_model = generator.generate_area_border_model()
            >>> print(border_model.columns)
            ['country_from', 'country_to', 'opposite_border', 'sorted_border', 'name_is_alphabetically_sorted']
            >>> 
            >>> # Access border relationships
            >>> for border_id, row in border_model.iterrows():
            ...     print(f"{border_id}: {row['country_from']} → {row['country_to']}")
        """
        borders = self._identify_borders()
        
        if not borders:
            return pd.DataFrame(
                columns=[
                    self.border_identifier,
                    self.source_area_identifier,
                    self.target_area_identifier,
                    self.OPPOSITE_BORDER_IDENTIFIER,
                    self.SORTED_BORDER_IDENTIFIER,
                    self.NAME_IS_ALPHABETICALLY_SORTED_IDENTIFIER,
                ]
            )
        
        border_data = []
        for area_from, area_to in borders:
            border_id = self.get_area_border_name(area_from, area_to)
            opposite_id = self.get_opposite_area_border_name(border_id)
            
            sorted_border = self.get_alphabetically_sorted_border(border_id)

            border_data.append({
                self.border_identifier: border_id,
                self.source_area_identifier: area_from,
                self.target_area_identifier: area_to,
                self.OPPOSITE_BORDER_IDENTIFIER: opposite_id,
                self.SORTED_BORDER_IDENTIFIER: sorted_border,
                self.NAME_IS_ALPHABETICALLY_SORTED_IDENTIFIER: sorted_border == border_id,
            })
        
        border_model_df = pd.DataFrame(border_data).set_index(self.border_identifier)

        return border_model_df

    def _identify_borders(self) -> set[tuple[str, str]]:
        """Identify borders from line topology.
        
        Analyzes line connectivity to find areas that are connected by
        lines, creating bidirectional border relationships.
        
        Returns:
            set: Set of (area_from, area_to) tuples representing directional borders.
                Includes both directions for each physical connection.
                
        Note:
            - Lines connecting nodes within the same area are ignored
            - Lines with nodes having None/NaN area assignments are ignored
            - Both directions (A→B and B→A) are included for each connection
        """
        borders = set()
        
        for _, line in self.line_model_df.iterrows():
            node_from = line[self.node_from_col]
            node_to = line[self.node_to_col]
            
            area_from = self.node_to_area_map.get(node_from)
            area_to = self.node_to_area_map.get(node_to)
            
            if area_from and area_to and area_from != area_to:
                borders.add((area_from, area_to))
                borders.add((area_to, area_from))
        
        return borders
    
    def _get_lines_for_border(self, area_from: str, area_to: str) -> list[str]:
        """Get all lines that cross a specific directional border.
        
        Args:
            area_from: Source area identifier
            area_to: Target area identifier
            
        Returns:
            list[str]: List of line identifiers that connect the specified areas
                in the given direction
                
        Example:

            >>> lines = generator._get_lines_for_border('DE', 'FR')
            >>> print(f"Lines from DE to FR: {lines}")
        """
        lines = []
        
        for line_id, line in self.line_model_df.iterrows():
            node_from = line[self.node_from_col]
            node_to = line[self.node_to_col]
            
            node_area_from = self.node_to_area_map.get(node_from)
            node_area_to = self.node_to_area_map.get(node_to)
            
            if node_area_from == area_from and node_area_to == area_to:
                lines.append(line_id)
        
        return lines
    
    def get_area_graph(self) -> nx.Graph:
        """Generate NetworkX graph representation of area connectivity.
        
        Creates an undirected graph where nodes represent areas and edges
        represent borders. This is useful for network analysis, path finding,
        and connectivity studies in multi-area energy systems.
        
        Returns:
            nx.Graph: Undirected graph with areas as nodes and borders as edges.
                Graph may contain multiple disconnected components if areas
                are not fully interconnected.
                
        Example:

            >>> graph = generator.get_area_graph()
            >>> print(f"Areas: {list(graph.nodes())}")
            >>> print(f"Borders: {list(graph.edges())}")
            >>> 
            >>> # Check connectivity
            >>> connected = nx.is_connected(graph)
            >>> print(f"All areas connected: {connected}")
        """
        graph = nx.Graph()
        borders = self._identify_borders()
        
        for area_from, area_to in borders:
            if not graph.has_edge(area_from, area_to):
                graph.add_edge(area_from, area_to)
        
        return graph
    
    def enhance_with_geometry(
        self, 
        border_model_df: pd.DataFrame,
        area_geometry_calculator: AreaBorderGeometryCalculator
    ) -> pd.DataFrame:
        """Enhance border model with geometric properties for visualization.
        
        Integrates with AreaBorderGeometryCalculator to add geometric information
        to borders, including representative points, directional angles, and
        line geometries. This enables advanced visualization of energy system borders.
        
        Args:
            border_model_df: Border model DataFrame to enhance
            area_geometry_calculator: Configured geometry calculator with area
                polygon data for geometric computations
                
        Returns:
            pd.DataFrame: Enhanced border model with additional geometric columns:
                - projection_point: Point for label/arrow placement
                - azimuth_angle: Directional angle in degrees
                - is_physical: Boolean indicating if border is physical (touching areas)
                - geo_line_string: LineString geometry representing the border
                
        Example:

            >>> # Setup geometry calculator with area polygons
            >>> geo_calc = AreaBorderGeometryCalculator(area_polygons_gdf)
            >>> 
            >>> # Enhance border model
            >>> enhanced_borders = generator.enhance_with_geometry(border_model, geo_calc)
            >>> print(enhanced_borders.columns)  # Includes geometric properties
            
        Note:
            Geometric enhancement may fail for some borders due to missing
            area geometries or calculation errors. Such failures are logged
            as warnings without stopping the overall process.
        """
        enhanced_df = border_model_df.copy()
        
        for border_id, border in border_model_df.iterrows():
            area_from = border[self.source_area_identifier]
            area_to = border[self.target_area_identifier]
            
            try:
                geometry_info = area_geometry_calculator.calculate_border_geometry(
                    area_from, area_to
                )

                enhanced_df.loc[border_id, self.PROJECTION_POINT_IDENTIFIER] = geometry_info[area_geometry_calculator.PROJECTION_POINT_IDENTIFIER]
                enhanced_df.loc[border_id, self.AZIMUTH_ANGLE_IDENTIFIER] = geometry_info[area_geometry_calculator.AZIMUTH_ANGLE_IDENTIFIER]
                enhanced_df.loc[border_id, self.BORDER_IS_PHYSICAL_IDENTIFIER] = geometry_info[area_geometry_calculator.BORDER_IS_PHYSICAL_IDENTIFIER]
                enhanced_df.loc[border_id, self.BORDER_LINE_STRING_IDENTIFIER] = geometry_info[area_geometry_calculator.BORDER_LINE_STRING_IDENTIFIER]

            except Exception as e:
                print(f"Warning: Could not calculate geometry for border {border_id} "
                      f"({area_from} → {area_to}): {e}")
        
        return enhanced_df


if __name__ == '__main__':
    # Comprehensive demonstration of AreaBorderModelGenerator capabilities
    
    # Create realistic European energy infrastructure data
    node_model_df = pd.DataFrame({
        'voltage_kv': [380, 380, 220, 380, 220, 150, 400, 220, 380, 150],
        'asset_type': ['transmission', 'transmission', 'distribution', 'transmission', 
                      'distribution', 'distribution', 'transmission', 'distribution',
                      'transmission', 'distribution'],
        'country': ['DE', 'DE', 'FR', 'FR', 'BE', 'BE', 'NL', 'NL', 'PL', 'PL'],
        'bidding_zone': ['DE_LU', 'DE_LU', 'FR', 'FR', 'BE', 'BE', 'NL', 'NL', 'PL', 'PL'],
        'capacity_mw': [2000, 1500, 800, 1200, 600, 400, 1800, 700, 1600, 500],
        'operator': ['TenneT', '50Hertz', 'RTE', 'RTE', 'Elia', 'Elia', 'TenneT', 'Stedin', 'PSE', 'PSE']
    }, index=['DE_T1', 'DE_T2', 'FR_D1', 'FR_T1', 'BE_D1', 'BE_D2', 'NL_T1', 'NL_D1', 'PL_T1', 'PL_D2'])
    
    # Realistic transmission line connections
    line_model_df = pd.DataFrame({
        'node_from': ['DE_T1', 'DE_T2', 'FR_T1', 'FR_T1', 'BE_D1', 'BE_D2', 'NL_T1', 'DE_T1', 'PL_T1'],
        'node_to': ['FR_T1', 'BE_D1', 'BE_D1', 'DE_T1', 'NL_T1', 'FR_D1', 'DE_T2', 'PL_T1', 'DE_T2'],
        'capacity_mw': [3000, 2200, 1800, 3200, 1500, 900, 2600, 2800, 2400],
        'length_km': [650, 320, 180, 650, 120, 280, 180, 580, 180],
        'voltage_kv': [380, 380, 220, 380, 220, 150, 400, 380, 380],
        'technology': ['AC', 'AC', 'AC', 'AC', 'AC', 'AC', 'DC', 'AC', 'AC']
    }, index=['L_DE_FR_1', 'L_DE_BE_1', 'L_FR_BE_1', 'L_FR_DE_1', 'L_BE_NL_1', 'L_BE_FR_1', 'L_NL_DE_1', 'L_DE_PL_1', 'L_PL_DE_1'])
    
    print("=== AREA BORDER MODEL GENERATOR DEMONSTRATION ===")
    print(f"Node model: {len(node_model_df)} nodes across {len(node_model_df['country'].unique())} countries")
    print(f"Line model: {len(line_model_df)} transmission lines")
    print(f"Countries: {sorted(node_model_df['country'].unique())}")
    print()
    
    # 1. Generate border model for countries
    print("1. COUNTRY-LEVEL BORDER MODEL")
    print("=" * 40)
    generator_country = AreaBorderModelGenerator(
        node_model_df,
        line_model_df,
        area_column='country',
        node_from_col='node_from',
        node_to_col='node_to'
    )
    
    border_model_country = generator_country.generate_area_border_model()
    print("Generated border model:")
    print(border_model_country)
    print()
    
    print("Border relationships:")
    for border_id, row in border_model_country.iterrows():
        opposite = row['opposite_border']
        sorted_border = row['sorted_border']
        is_sorted = row['name_is_alphabetically_sorted']
        print(f"  {border_id}: {row['country_from']} → {row['country_to']} "
              f"(opposite: {opposite}, canonical: {sorted_border}, sorted: {is_sorted})")
    print()
    
    # 2. Generate border model for bidding zones
    print("2. BIDDING ZONE BORDER MODEL")
    print("=" * 40)
    generator_bz = AreaBorderModelGenerator(
        node_model_df,
        line_model_df,
        area_column='bidding_zone',
        node_from_col='node_from',
        node_to_col='node_to'
    )
    
    border_model_bz = generator_bz.generate_area_border_model()
    print(f"Bidding zone borders: {len(border_model_bz)} directional relationships")
    print(border_model_bz[['bidding_zone_from', 'bidding_zone_to']])
    print()
    
    # 3. Network connectivity analysis
    print("3. NETWORK CONNECTIVITY ANALYSIS")
    print("=" * 40)
    
    # Country-level connectivity
    country_graph = generator_country.get_area_graph()
    print(f"Country connectivity graph:")
    print(f"  Nodes (countries): {sorted(list(country_graph.nodes()))}")
    print(f"  Edges (borders): {sorted(list(country_graph.edges()))}")
    print(f"  Is connected: {nx.is_connected(country_graph)}")
    print(f"  Number of components: {nx.number_connected_components(country_graph)}")
    print()
    
    # Analyze connectivity paths
    if nx.is_connected(country_graph):
        # Find shortest paths between all country pairs
        all_pairs_paths = dict(nx.all_pairs_shortest_path(country_graph))
        print("Shortest paths between countries:")
        for source in sorted(country_graph.nodes()):
            for target in sorted(country_graph.nodes()):
                if source < target:  # Avoid duplicates
                    path = all_pairs_paths[source][target]
                    print(f"  {source} → {target}: {' → '.join(path)} (length: {len(path)-1})")
        print()
    
    # 4. Demonstrate naming conventions
    print("4. NAMING CONVENTIONS DEMONSTRATION")
    print("=" * 40)
    
    conventions = AreaBorderNamingConventions('market_zone')
    
    example_pairs = [('DE_North', 'DE_South'), ('FR_West', 'ES_North'), ('IT_North', 'CH_South')]
    
    print("Border naming examples:")
    for area_from, area_to in example_pairs:
        border_name = conventions.get_area_border_name(area_from, area_to)
        opposite = conventions.get_opposite_area_border_name(border_name)
        canonical = conventions.get_alphabetically_sorted_border(border_name)
        
        print(f"  Original: {border_name}")
        print(f"  Opposite: {opposite}")
        print(f"  Canonical: {canonical}")
        print()
