from typing import Hashable
from dataclasses import dataclass
from shapely.geometry import Point
import networkx as nx


@dataclass
class InterpolationConfig:
    move_percentage: float = 0.01
    max_iterations: int = 100
    tolerance: float = 1e-5


class GraphNodeCoordinateInterpolator:
    """
    Provides a flexible, iterative approach to interpolate missing node coordinates in 2D spatial networks
    """
    def __init__(
            self,
            edges: list[tuple[Hashable, Hashable]],
            known_node_positions: dict[Hashable, Point],
            config: InterpolationConfig = InterpolationConfig()
    ):
        self._graph = nx.Graph()
        self._graph.add_edges_from(edges)
        self._known_node_positions = known_node_positions
        self._all_node_positions: dict[Hashable, Point] = known_node_positions.copy()
        self._has_computed: bool = False
        self._config = config

    def compute(self):
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
                        pos.x + move_percentage * movement_vector.x,
                        pos.y + move_percentage * movement_vector.y
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
                    movement_vector.y + (self._all_node_positions[adj].y - pos.y)
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
        return self._all_node_positions[node]

    def get_all_node_positions(self) -> dict[Hashable, Point]:
        return self._all_node_positions


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    test_edges = [('A', 'B'), ('B', 'C'), ('C', 'D'), ('C', 'F'), ('F', 'G'), ('B', 'H'), ('H', 'I'), ('I', 'J'), ('J', 'K')]
    given_positions = {'A': Point(0, 0), 'D': Point(1, 2), 'F': Point(2, 1), 'K': Point(0, 1)}

    inter = GraphNodeCoordinateInterpolator(test_edges, given_positions)
    inter.compute()
    all_positions = inter.get_all_node_positions()
    print("Computed positions:", all_positions)

    def plot_graph():
        pos = {node: (point.x, point.y) for node, point in all_positions.items()}
        nx.draw(
            inter._graph,
            pos,
            with_labels=True,
            node_size=500,
            node_color='lightblue',
            font_size=10,
            font_weight='bold',
            edge_color='gray'
        )
        plt.show()
    plot_graph()
