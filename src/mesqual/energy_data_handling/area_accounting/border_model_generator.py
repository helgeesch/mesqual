"""Border models: the borders between areas from the line topology, with naming conventions and geometry."""

from __future__ import annotations

import logging
from typing import Tuple
import pandas as pd
import networkx as nx

from mesqual.energy_data_handling.area_accounting.area_membership import AreaMembership
from mesqual.energy_data_handling.area_accounting.border_model_geometry_calculator import AreaBorderGeometryCalculator

logger = logging.getLogger(__name__)


class AreaBorderNamingConventions:
    """Standardized naming conventions for energy system area borders.

    This class provides consistent naming patterns for borders between energy
    system areas (countries, bidding zones, market regions). It ensures
    standardized naming across different analysis workflows and supports
    bidirectional relationship management.

    It gives:

    - Border names in both directions (A - B and B - A) and the opposite
      border of a border.
    - Canonical border names: the area names in alphabetical order.
    - The column names of the border model, configurable per instance.

    Attributes:
        area_column (str): Name of the area column.
        border_identifier (str): Name of the border column. Default:
            '{area_column}_border'.
        source_area_identifier (str): Name of the source area column. Default:
            `AreaMembership.AREA_FROM` ('area_from').
        target_area_identifier (str): Name of the target area column. Default:
            `AreaMembership.AREA_TO` ('area_to').
        JOIN_AREA_NAMES_BY (str): Separator for area names in border identifiers
        OPPOSITE_BORDER_IDENTIFIER (str): Column name for opposite border references
        CANONICAL_BORDER_IDENTIFIER (str): Column name for the canonical border
            of each border: the border with the area names in alphabetical order
        IS_CANONICAL_IDENTIFIER (str): Column name for the boolean that is True
            where the border is its canonical border

    Examples:
        >>> conventions = AreaBorderNamingConventions('country')
        >>> conventions.get_area_border_name('C1', 'C2')
        'C1 - C2'
        >>> conventions.get_opposite_area_border_name('C1 - C2')
        'C2 - C1'
        >>> conventions.get_canonical_border('C2 - C1')
        'C1 - C2'
        >>> conventions.border_identifier
        'country_border'
    """

    JOIN_AREA_NAMES_BY = ' - '
    OPPOSITE_BORDER_IDENTIFIER = 'opposite_border'
    CANONICAL_BORDER_IDENTIFIER = 'canonical_border'
    IS_CANONICAL_IDENTIFIER = 'is_canonical'
    PROJECTION_POINT_IDENTIFIER = AreaBorderGeometryCalculator.PROJECTION_POINT_IDENTIFIER
    AZIMUTH_ANGLE_IDENTIFIER = AreaBorderGeometryCalculator.AZIMUTH_ANGLE_IDENTIFIER
    BORDER_AREAS_TOUCH_IDENTIFIER = AreaBorderGeometryCalculator.BORDER_AREAS_TOUCH_IDENTIFIER
    BORDER_LINE_STRING_IDENTIFIER = AreaBorderGeometryCalculator.BORDER_LINE_STRING_IDENTIFIER

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
            source_area_identifier: Name of the source area column.
                Defaults to `AreaMembership.AREA_FROM` ('area_from').
            target_area_identifier: Name of the target area column.
                Defaults to `AreaMembership.AREA_TO` ('area_to').
        """
        self.area_column = area_column
        self.border_identifier = border_identifier or f'{area_column}_border'
        self.source_area_identifier = source_area_identifier or AreaMembership.AREA_FROM
        self.target_area_identifier = target_area_identifier or AreaMembership.AREA_TO

    def get_area_border_name(self, area_from: str, area_to: str) -> str:
        """Generate standardized border name from source and target areas.

        Args:
            area_from: Source area identifier (e.g., 'DE', 'FR_North')
            area_to: Target area identifier (e.g., 'FR', 'DE_South')

        Returns:
            str: Formatted border name using the configured separator (e.g. 'DE - FR', 'FR_North - DE_South')
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

        Examples:
            >>> AreaBorderNamingConventions('country').decompose_area_border_name_to_areas('C1 - C2')
            ('C1', 'C2')
        """
        area_from, area_to = border_name.split(self.JOIN_AREA_NAMES_BY)
        return area_from, area_to

    def get_opposite_area_border_name(self, border_name: str) -> str:
        """Generate the opposite direction border name.

        Args:
            border_name: Original border name (e.g., 'DE - FR')

        Returns:
            str: Opposite direction border name (e.g., 'FR - DE')
        """
        area_from, area_to = self.decompose_area_border_name_to_areas(border_name)
        return self.get_area_border_name(area_to, area_from)

    def get_canonical_border(self, border_name: str) -> str:
        """Return the canonical border of a border.

        The canonical border has the area names in alphabetical order. A border
        and its opposite border have the same canonical border. Use it to
        identify a pair of borders regardless of direction, or to match borders
        of opposite direction.

        Args:
            border_name: Border name in any direction (e.g., 'FR - DE' or 'DE - FR')

        Returns:
            str: Canonical border name (e.g., 'DE - FR')
        """
        area_from, area_to = self.decompose_area_border_name_to_areas(border_name)
        return self.get_area_border_name(*list(sorted([area_from, area_to])))


class AreaBorderModelGenerator(AreaBorderNamingConventions, AreaMembership):
    """Border model from the lines and the area of each node.

    `line_areas` maps the two ends of each line to their areas
    (`AreaMembership.edge_areas`), in the columns `source_area_identifier`
    and `target_area_identifier` (default `area_from` and `area_to`). A line
    between two different areas gives two borders, one per direction. A line
    inside one area, or with an end without area, gives no border. The
    borders follow the order of the lines. Each border has its source and
    target area, its opposite border, its canonical border and
    `is_canonical` (`AreaBorderNamingConventions`).

    Attributes:
        line_model_df (pd.DataFrame): Line model, one row per line.
        node_model_df (pd.DataFrame): Node model with the area column.
        node_from_col (str): Column of `line_model_df` with the first node.
        node_to_col (str): Column of `line_model_df` with the second node.
        line_areas (pd.DataFrame): The source and target area per line.

    Examples:
        >>> generator = AreaBorderModelGenerator(
        ...     dataset.fetch('Node.Model'), dataset.fetch('Line.Model'), 'country', 'node_from', 'node_to'
        ... )
        >>> generator.generate_area_border_model()
                       area_from area_to opposite_border canonical_border  is_canonical
        country_border
        C1 - C2               C1      C2         C2 - C1          C1 - C2          True
        C2 - C1               C2      C1         C1 - C2          C1 - C2         False
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
            source_area_identifier: Name of the source area column.
                Defaults to `AreaMembership.AREA_FROM` ('area_from').
            target_area_identifier: Name of the target area column.
                Defaults to `AreaMembership.AREA_TO` ('area_to').

        Raises:
            ValueError: If required columns are not found in input DataFrames,
                or if `node_model_df` has a node id more than once.
        """
        super().__init__(area_column, border_identifier, source_area_identifier, target_area_identifier)
        self.line_model_df = line_model_df
        self.node_model_df = node_model_df
        self.node_from_col = node_from_col
        self.node_to_col = node_to_col

        self._validate_inputs()
        self.line_areas = self.edge_areas(
            line_model_df,
            node_model_df[area_column],
            node_from_col,
            node_to_col,
            self.source_area_identifier,
            self.target_area_identifier,
        )

    def _validate_inputs(self):
        self._check_area_column()
        if self.node_from_col not in self.line_model_df.columns:
            raise ValueError(
                f"Source node column '{self.node_from_col}' not found in line_model_df. "
                f'Available columns: {list(self.line_model_df.columns)}'
            )
        if self.node_to_col not in self.line_model_df.columns:
            raise ValueError(
                f"Target node column '{self.node_to_col}' not found in line_model_df. "
                f'Available columns: {list(self.line_model_df.columns)}'
            )

    def generate_area_border_model(self) -> pd.DataFrame:
        """Generate the border model.

        Analyzes transmission line topology to identify borders between areas,
        creating a DataFrame with directional relationships,
        naming conventions, and reference data for further analysis.

        The generated model includes:

        - Border identifiers in both directions (A - B and B - A).
        - Source and target area columns.
        - The opposite border of each border.
        - `canonical_border`: the border of the pair with the area names in alphabetical order.
        - `is_canonical`: True where the border is its canonical border.

        Returns:
            pd.DataFrame: Border model indexed by border identifiers (index
                name `border_identifier`). The rows follow the order of the
                lines: first the area pairs in line direction, then the reverse
                pairs. Without borders, it has no rows and the same index name
                and columns.
        """
        border_data = []
        for area_from, area_to in self._identify_borders():
            border_id = self.get_area_border_name(area_from, area_to)
            opposite_id = self.get_opposite_area_border_name(border_id)

            canonical_border = self.get_canonical_border(border_id)

            border_data.append(
                {
                    self.border_identifier: border_id,
                    self.source_area_identifier: area_from,
                    self.target_area_identifier: area_to,
                    self.OPPOSITE_BORDER_IDENTIFIER: opposite_id,
                    self.CANONICAL_BORDER_IDENTIFIER: canonical_border,
                    self.IS_CANONICAL_IDENTIFIER: canonical_border == border_id,
                }
            )

        columns = [
            self.border_identifier,
            self.source_area_identifier,
            self.target_area_identifier,
            self.OPPOSITE_BORDER_IDENTIFIER,
            self.CANONICAL_BORDER_IDENTIFIER,
            self.IS_CANONICAL_IDENTIFIER,
        ]
        border_model_df = pd.DataFrame(border_data, columns=columns).set_index(self.border_identifier)
        border_model_df = border_model_df.astype({self.IS_CANONICAL_IDENTIFIER: bool})

        return border_model_df

    def _identify_borders(self) -> list[tuple]:
        """Return the `(area_from, area_to)` pairs of `line_areas`, in both directions, in line order.

        A line with both ends in one area, or with an end without an area, gives no border.
        """
        pairs = self.line_areas[self.crosses_border(self.line_areas)]
        area_from, area_to = pairs[self.source_area_identifier], pairs[self.target_area_identifier]
        return list(dict.fromkeys([*zip(area_from, area_to), *zip(area_to, area_from)]))

    def get_area_graph(self) -> nx.Graph:
        """Generate NetworkX graph representation of area connectivity.

        Creates an undirected graph where nodes represent areas and edges
        represent borders. This is useful for network analysis, path finding,
        and connectivity studies in multi-area energy systems.

        Returns:
            nx.Graph: Undirected graph with areas as nodes and borders as edges.
                Graph may contain multiple disconnected components if areas
                are not fully interconnected.

        Examples:
            >>> generator = AreaBorderModelGenerator(
            ...     dataset.fetch('Node.Model'), dataset.fetch('Line.Model'), 'country', 'node_from', 'node_to'
            ... )
            >>> list(generator.get_area_graph().edges())
            [('C1', 'C2')]
        """
        graph = nx.Graph()
        graph.add_edges_from(self._identify_borders())
        return graph

    def enhance_with_geometry(
        self, border_model_df: pd.DataFrame, area_geometry_calculator: AreaBorderGeometryCalculator
    ) -> pd.DataFrame:
        """Enhance border model with geometric properties for visualization.

        Uses the AreaBorderGeometryCalculator to add a representative point,
        a direction angle and a line geometry to each border.

        Args:
            border_model_df: Border model DataFrame to enhance
            area_geometry_calculator: Configured geometry calculator with area
                polygon data for geometric computations

        Returns:
            pd.DataFrame: The border model with the columns `projection_point`
                (point for labels and arrows), `azimuth_angle` (direction in
                degrees), `areas_touch` (True when the areas touch or overlap)
                and `geo_line_string` (the border line).

        Examples:
            >>> import geopandas as gpd
            >>> from mesqual.energy_data_handling.area_accounting import AreaBorderGeometryCalculator
            >>> generator = AreaBorderModelGenerator(
            ...     dataset.fetch('Node.Model'), dataset.fetch('Line.Model'), 'country', 'node_from', 'node_to'
            ... )
            >>> countries = gpd.GeoDataFrame(dataset.fetch('Country.Model'), crs='EPSG:4326')
            >>> borders = generator.enhance_with_geometry(generator.generate_area_border_model(), AreaBorderGeometryCalculator(countries))
            >>> borders[['azimuth_angle', 'areas_touch']]
                            azimuth_angle  areas_touch
            country_border
            C1 - C2                  90.0         True
            C2 - C1                 270.0         True

        Note:
            The geometry may fail for some borders, for example for a missing
            area geometry. Such a border gets NaN in the geometry columns, and
            the failure is logged at `WARNING`. The other borders get their
            geometry. The result always has the geometry columns.
        """
        columns = list(area_geometry_calculator.GEOMETRY_COLUMNS)
        rows = {}
        for border_id, border in border_model_df.iterrows():
            area_from = border[self.source_area_identifier]
            area_to = border[self.target_area_identifier]
            try:
                geometry_info = area_geometry_calculator.calculate_border_geometry(area_from, area_to)
            except Exception as e:
                logger.warning(
                    'Could not calculate the geometry of border %s (%s to %s): %s',
                    border_id,
                    area_from,
                    area_to,
                    e,
                )
                continue
            rows[border_id] = [geometry_info[column] for column in columns]
        geometry = pd.DataFrame.from_dict(rows, orient='index', columns=columns).reindex(border_model_df.index)
        return border_model_df.drop(columns=columns, errors='ignore').join(geometry)
