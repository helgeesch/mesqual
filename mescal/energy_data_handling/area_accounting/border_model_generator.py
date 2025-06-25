from typing import Tuple
import pandas as pd
import networkx as nx

from mescal.energy_data_handling.area_accounting.border_model_geometry_calculator import AreaBorderGeometryCalculator


class AreaBorderNamingConventions:

    JOIN_AREA_NAMES_BY = ' - '
    SOURCE_AREA_IDENTIFIER_PREFIX = ''
    TARGET_AREA_IDENTIFIER_PREFIX = ''
    SOURCE_AREA_IDENTIFIER_SUFFIX = '_from'
    TARGET_AREA_IDENTIFIER_SUFFIX = '_to'
    OPPOSITE_BORDER_IDENTIFIER = 'opposite_border'
    SORTED_BORDER_IDENTIFIER = 'sorted_border'
    NAME_IS_ALPHABETICALLY_SORTED_IDENTIFIER = 'name_is_alphabetically_sorted'
    PROJECTION_POINT_IDENTIFIER = 'projection_point'
    PROJECTION_ANGLE_IDENTIFIER = 'projection_angle'
    BORDER_IS_PHYSICAL_IDENTIFIER = 'is_physical'
    BORDER_LINE_STRING_IDENTIFIER = 'geo_line_string'

    def __init__(
            self,
            area_column: str,
            border_identifier: str = None,
            source_area_identifier: str = None,
            target_area_identifier: str = None,
    ):
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
        return f'{area_from}{self.JOIN_AREA_NAMES_BY}{area_to}'

    def decompose_area_border_name_to_areas(self, border_name: str) -> Tuple[str, str]:
        area_from, area_to = border_name.split(self.JOIN_AREA_NAMES_BY)
        return area_from, area_to

    def get_opposite_area_border_name(self, border_name: str):
        area_from, area_to = self.decompose_area_border_name_to_areas(border_name)
        return self.get_area_border_name(area_to, area_from)

    def get_alphabetically_sorted_border(self, border_name: str) -> str:
        area_from, area_to = self.decompose_area_border_name_to_areas(border_name)
        return self.get_area_border_name(*list(sorted([area_from, area_to])))


class AreaBorderModelGenerator(AreaBorderNamingConventions):
    """Identifies and generates area border model from line topology and node-area mapping."""

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
        super().__init__(area_column, border_identifier, source_area_identifier, target_area_identifier)
        self.line_model_df = line_model_df
        self.node_model_df = node_model_df
        self.node_from_col = node_from_col
        self.node_to_col = node_to_col

        self._validate_inputs()
        self.node_to_area_map = self._create_node_to_area_map()

    def _validate_inputs(self):
        if self.area_column not in self.node_model_df.columns:
            raise ValueError(f"Column '{self.area_column}' not found in node_model_df")
        if self.node_from_col not in self.line_model_df.columns:
            raise ValueError(f"Column '{self.node_from_col}' not found in line_model_df")
        if self.node_to_col not in self.line_model_df.columns:
            raise ValueError(f"Column '{self.node_to_col}' not found in line_model_df")

    def _create_node_to_area_map(self) -> dict:
        return self.node_model_df[self.area_column].to_dict()
    
    def generate_area_border_model(self) -> pd.DataFrame:
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
        """Returns a networkx graph of areas connected by borders."""
        graph = nx.Graph()
        borders = self._identify_borders()
        
        for area_from, area_to in borders:
            if not graph.has_edge(area_from, area_to):
                graph.add_edge(area_from, area_to)
        
        return graph
    
    def enhance_with_geometry(
        self, 
        border_model_df: pd.DataFrame,
        area_geometry_calculator: 'AreaBorderGeometryCalculator'
    ) -> pd.DataFrame:
        """Enhance border model with geometric properties.
        
        Args:
            border_model_df: Border model DataFrame to enhance
            area_geometry_calculator: Geometry calculator with area geometries
            
        Returns:
            Enhanced DataFrame with geometry, projection_point, and projection_angle
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
                enhanced_df.loc[border_id, self.PROJECTION_ANGLE_IDENTIFIER] = geometry_info[area_geometry_calculator.PROJECTION_ANGLE_IDENTIFIER]
                enhanced_df.loc[border_id, self.BORDER_IS_PHYSICAL_IDENTIFIER] = geometry_info[area_geometry_calculator.BORDER_IS_PHYSICAL_IDENTIFIER]
                enhanced_df.loc[border_id, self.BORDER_LINE_STRING_IDENTIFIER] = geometry_info[area_geometry_calculator.BORDER_LINE_STRING_IDENTIFIER]

            except Exception as e:
                print(f"Warning: Could not calculate geometry for {border_id}: {e}")
        
        return enhanced_df


if __name__ == '__main__':
    # Create dummy data
    node_model_df = pd.DataFrame({
        'country': ['DE', 'DE', 'FR', 'FR', 'BE', 'NL'],
        'bidding_zone': ['DE_LU', 'DE_LU', 'FR', 'FR', 'BE', 'NL']
    }, index=['DE1', 'DE2', 'FR1', 'FR2', 'BE1', 'NL1'])
    
    line_model_df = pd.DataFrame({
        'node_from': ['DE1', 'DE2', 'FR1', 'FR2', 'BE1'],
        'node_to': ['FR1', 'BE1', 'BE1', 'DE1', 'NL1'],
        'capacity': [1000, 800, 1200, 600, 900]
    }, index=['L1', 'L2', 'L3', 'L4', 'L5'])
    
    # Generate border model for countries
    generator = AreaBorderModelGenerator(
        node_model_df,
        line_model_df,
        area_column='country',
        node_from_col='node_from',
        node_to_col='node_to',
    )
    border_model = generator.generate_area_border_model()
    print("Area Border Model:")
    print(border_model)
    print()
    
    # Get area connectivity graph
    graph = generator.get_area_graph()
    print("Connected areas:", list(graph.edges()))