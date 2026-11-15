"""Area membership of nodes and edges: the one rule for which node belongs to which area."""

import pandas as pd


class AreaMembership:
    """Common parent of the area calculators and the area and border model generators: the area of each node.

    A node has an area if `has_area` is `True` for the value in its area
    column. A node without an area belongs to no area and is on no border.
    `edge_areas` maps the two ends of each edge to their areas with the same
    rule. A subclass sets `node_model_df` and `area_column`.

    Attributes:
        node_model_df: Node model, one row per node.
        area_column: Column of `node_model_df` with the area of each node.
    """

    AREA_FROM = 'area_from'
    AREA_TO = 'area_to'

    node_model_df: pd.DataFrame
    area_column: str

    @staticmethod
    def has_area(areas: pd.Series) -> pd.Series:
        """Return `True` per object whose area is set: neither `NaN` nor `None` nor an empty string.

        Examples:
            >>> AreaMembership.has_area(pd.Series({'n1': 'A', 'n2': None, 'n3': ''})).tolist()
            [True, False, False]
        """
        return areas.notna() & (areas != '')

    @classmethod
    def edge_areas(
        cls,
        edges: pd.DataFrame,
        area_of_node: pd.Series,
        end_from: str,
        end_to: str,
        area_from: str = AREA_FROM,
        area_to: str = AREA_TO,
    ) -> pd.DataFrame:
        """Return the area of the first end and the area of the second end per edge.

        Args:
            edges: The edge model, one row per edge.
            area_of_node: The area of each node, indexed by node.
            end_from: The column of `edges` with the first end.
            end_to: The column of `edges` with the second end.
            area_from: The name of the result column with the area of the first end.
            area_to: The name of the result column with the area of the second end.

        Returns:
            One row per edge with the two columns `area_from` and `area_to`,
                in this order, of dtype `object`, so that area ids keep their type.
                An end without area, or at a node that is not in `area_of_node`,
                gives `NaN`.

        Raises:
            ValueError: If `area_of_node` has a node id more than once.

        Examples:
            >>> nodes, lines = dataset.fetch('Node.Model'), dataset.fetch('Line.Model')
            >>> AreaMembership.edge_areas(lines, nodes['country'], 'node_from', 'node_to')
                 area_from area_to
            Line
            L1          C1      C2
            L2          C2      C1
            L3          C1      C1
            L4          C1      C2
            L5          C2      C2
        """
        duplicates = area_of_node.index[area_of_node.index.duplicated()].unique()
        if len(duplicates):
            raise ValueError(f'Node ids appear more than once in the node areas: {list(duplicates)}')
        areas = area_of_node[cls.has_area(area_of_node)].to_dict()
        return pd.DataFrame(
            {
                area_from: [areas.get(node, float('nan')) for node in edges[end_from]],
                area_to: [areas.get(node, float('nan')) for node in edges[end_to]],
            },
            index=edges.index,
            dtype=object,
        )

    @staticmethod
    def crosses_border(edge_areas: pd.DataFrame) -> pd.Series:
        """Return `True` per edge that is on a border: both areas are set and differ.

        An edge from area A to area B is on the border A - B, in border
        direction, and on the border B - A, against it.

        Args:
            edge_areas: Two columns per edge, as `edge_areas` gives them: the
                first with the area of the from end, the second with the area
                of the to end.
        """
        area_from, area_to = (edge_areas[column] for column in edge_areas.columns)
        return area_from.notna() & area_to.notna() & (area_from != area_to)

    def get_node_areas(self) -> pd.Series:
        """Return the area of each node that has one, indexed by node.

        Raises:
            ValueError: If `area_column` is not a column of `node_model_df`.
        """
        self._check_area_column()
        areas = self.node_model_df[self.area_column]
        return areas[self.has_area(areas)]

    def _check_area_column(self):
        if self.area_column not in self.node_model_df.columns:
            raise ValueError(
                f"Area column '{self.area_column}' not found in node_model_df. "
                f'Available columns: {list(self.node_model_df.columns)}'
            )
