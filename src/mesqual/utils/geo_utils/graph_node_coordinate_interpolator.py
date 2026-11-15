"""Positions of graph nodes estimated from the positions of their neighbours."""

from typing import Hashable
from dataclasses import dataclass
from shapely.geometry import Point
import networkx as nx


@dataclass
class InterpolationConfig:
    """Settings of `GraphNodeCoordinateInterpolator`.

    Attributes:
        move_percentage: After the iterations, each node that shares its
            position with another node moves by this share of the vector to
            the mean of its neighbours at other positions.
        max_iterations: Maximum number of iterations. Each iteration moves each
            node without known position to the mean position of its neighbours.
        tolerance: The iterations stop when each node moves less than this distance.
    """

    move_percentage: float = 0.01
    max_iterations: int = 100
    tolerance: float = 1e-5


class GraphNodeCoordinateInterpolator:
    """Estimates the positions of graph nodes without known position from the positions of their neighbours.

    `compute` places each node without known position at the center of the
    bounding box of the known positions, then moves it in iterations to the
    mean position of its neighbours (see `InterpolationConfig`). After the
    iterations, each node that shares its position with another node, known
    nodes included, moves by `move_percentage` toward the mean of its
    neighbours at other positions. Nodes with the same neighbours stay
    together. A component without a known node stays at the center of the
    bounding box.

    Args:
        edges: Node pairs of the graph.
        known_node_positions: Known position per node. The iterations do not
            change them. The dict itself is not changed.
        config: Settings of the iterations.

    Examples:
        >>> from shapely.geometry import Point
        >>> interpolator = GraphNodeCoordinateInterpolator(
        ...     [('A', 'B'), ('B', 'C')], {'A': Point(0, 0), 'C': Point(2, 0)}
        ... )
        >>> interpolator.compute()
        >>> interpolator.get_position_for_node('B')
        <POINT (1 0)>
    """

    def __init__(
        self,
        edges: list[tuple[Hashable, Hashable]],
        known_node_positions: dict[Hashable, Point],
        config: InterpolationConfig = InterpolationConfig(),
    ):
        self._graph = nx.Graph()
        self._graph.add_edges_from(edges)
        self._known_node_positions = known_node_positions
        self._all_node_positions: dict[Hashable, Point] = known_node_positions.copy()
        self._has_computed: bool = False
        self._config = config

    def compute(self):
        """Compute the positions of all nodes. Before this call, only the known positions are available."""
        self._initialize_unknown_positions()
        self._iterate_to_find_positions()
        self._resolve_node_overlaps()
        self._has_computed = True

    def _initialize_unknown_positions(self):
        min_x, max_x, min_y, max_y = self._get_known_positions_bounding_box()
        for node in self._graph.nodes:
            if node not in self._all_node_positions:
                self._all_node_positions[node] = Point((min_x + max_x) / 2, (min_y + max_y) / 2)

    def _iterate_to_find_positions(self):
        for _ in range(self._config.max_iterations):
            max_change = 0
            new_positions = self._all_node_positions.copy()
            for node in self._graph.nodes:
                if node not in self._known_node_positions:
                    new_pos, change = self._compute_average_position(node)
                    new_positions[node] = new_pos
                    max_change = max(max_change, change)
            self._all_node_positions = new_positions
            if max_change < self._config.tolerance:
                break

    def _compute_average_position(self, node: Hashable) -> tuple[Point, float]:
        neighbors = list(self._graph.neighbors(node))
        avg_x = sum(self._all_node_positions[neighbor].x for neighbor in neighbors) / len(neighbors)
        avg_y = sum(self._all_node_positions[neighbor].y for neighbor in neighbors) / len(neighbors)
        new_pos = Point(avg_x, avg_y)
        change = self._all_node_positions[node].distance(new_pos)
        return new_pos, change

    def _resolve_node_overlaps(self):
        positions = self._all_node_positions
        move_percentage = self._config.move_percentage
        node_positions = {}
        for node, pos in positions.items():
            node_positions.setdefault(pos, []).append(node)
        for pos, nodes in node_positions.items():
            if len(nodes) > 1:
                for node in nodes:
                    movement_vector = self._calculate_movement_vector(node, pos)
                    new_pos = Point(
                        pos.x + move_percentage * movement_vector.x, pos.y + move_percentage * movement_vector.y
                    )
                    positions[node] = new_pos
        self._all_node_positions = positions

    def _calculate_movement_vector(self, node, pos):
        adjacent_nodes = list(self._graph.neighbors(node))
        movement_vector = Point(0, 0)
        count = 0
        for adj in adjacent_nodes:
            if self._all_node_positions[adj] != pos:
                movement_vector = Point(
                    movement_vector.x + (self._all_node_positions[adj].x - pos.x),
                    movement_vector.y + (self._all_node_positions[adj].y - pos.y),
                )
                count += 1
        if count > 0:
            movement_vector = Point(movement_vector.x / count, movement_vector.y / count)
        return movement_vector

    def _get_known_positions_bounding_box(self) -> tuple[float, float, float, float]:
        min_x = min(point.x for point in self._known_node_positions.values())
        max_x = max(point.x for point in self._known_node_positions.values())
        min_y = min(point.y for point in self._known_node_positions.values())
        max_y = max(point.y for point in self._known_node_positions.values())
        return min_x, max_x, min_y, max_y

    def get_position_for_node(self, node: Hashable) -> Point:
        """Return the position of a node.

        Raises:
            KeyError: The node has no position, for example a node without
                known position before `compute`.
        """
        return self._all_node_positions[node]

    def get_all_node_positions(self) -> dict[Hashable, Point]:
        """Return the position per node. The dict is the internal one until the next `compute`, which replaces it."""
        return self._all_node_positions
