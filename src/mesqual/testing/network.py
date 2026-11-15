"""Synthetic network: countries, nodes, lines, generators and demands."""

from __future__ import annotations

import pandas as pd
from shapely.geometry import Point, box

from mesqual.testing.scenario import DummyScenario


class DummyNetwork:
    """Model tables of a synthetic network.

    Countries are squares side by side from west to east. Nodes are placed round
    robin in the countries, and each node lies inside the square of its country.
    Node k connects by line to node k-1 and to node k-2, in this order. Generator
    k and demand k sit at node k // 2, two generators and two demands per node.
    More nodes, lines, generators or demands only append rows, so existing rows
    stay the same.

    Args:
        scenario: The scenario spec.
    """

    COUNTRY_SIZE_DEG = 4.0
    COUNTRY_ORIGIN = (5.0, 46.0)
    FUELS = ('nuclear', 'coal', 'gas', 'wind', 'solar')
    SECTORS = ('industry', 'households', 'services')
    LINE_CAPACITY_MW = 1000.0
    GENERATOR_CAPACITY_MW = 800.0
    ORPHAN_NODE = 'N0'
    GENERATORS_PER_NODE = 2
    DEMANDS_PER_NODE = 2

    def __init__(self, scenario: DummyScenario):
        self._scenario = scenario
        tables = {
            'countries': self._build_countries(),
            'nodes': self._build_nodes(),
            'lines': self._build_lines(),
            'generators': self._build_generators(),
            'demands': self._build_demands(),
        }
        tables = self._remove_objects(tables, scenario.removed_objects)
        self.tables: dict[str, pd.DataFrame] = self._apply_text_changes(tables, scenario.text_changes)

    @property
    def countries(self) -> pd.DataFrame:
        """Country table, index 'Country', columns name and geometry (square polygons)."""
        return self.tables['countries']

    @property
    def nodes(self) -> pd.DataFrame:
        """Node table, index 'Node', columns country, voltage, base_price and location (points)."""
        return self.tables['nodes']

    @property
    def lines(self) -> pd.DataFrame:
        """Line table, index 'Line', columns node_from, node_to, capacity and voltage."""
        return self.tables['lines']

    @property
    def generators(self) -> pd.DataFrame:
        """Generator table, index 'Generator', columns node, fuel and capacity."""
        return self.tables['generators']

    @property
    def demands(self) -> pd.DataFrame:
        """Demand table, index 'Demand', columns node, sector and base_demand."""
        return self.tables['demands']

    def _build_countries(self) -> pd.DataFrame:
        n = self._scenario.n_countries
        x0, y0 = self.COUNTRY_ORIGIN
        size = self.COUNTRY_SIZE_DEG
        return pd.DataFrame(
            {
                'name': [f'Country {i + 1}' for i in range(n)],
                'geometry': [box(x0 + i * size, y0, x0 + (i + 1) * size, y0 + size) for i in range(n)],
            },
            index=pd.Index([f'C{i + 1}' for i in range(n)], name='Country'),
        )

    def _build_nodes(self) -> pd.DataFrame:
        s = self._scenario
        rows = [self._node_row(i, i % s.n_countries) for i in range(s.n_nodes)]
        index = [f'N{i + 1}' for i in range(s.n_nodes)]
        if s.node_without_country:
            rows.append(self._node_row(-1, None))
            index.append(self.ORPHAN_NODE)
        return pd.DataFrame(rows, index=pd.Index(index, name='Node'))

    def _node_row(self, i: int, country_number: int | None) -> dict:
        x0, y0 = self.COUNTRY_ORIGIN
        size = self.COUNTRY_SIZE_DEG
        x_country = x0 + country_number * size if country_number is not None else x0 - size
        rank = i // self._scenario.n_countries
        return {
            'country': f'C{country_number + 1}' if country_number is not None else None,
            'voltage': 380.0 if i % 2 == 0 else 220.0,
            'base_price': 40.0 + 5.0 * i,
            'location': Point(
                x_country + size * (0.1 + 0.8 * (rank * 0.618 % 1)), y0 + size * (0.1 + 0.8 * (rank * 0.382 % 1))
            ),
        }

    def _build_lines(self) -> pd.DataFrame:
        pairs = self._line_pairs()
        return pd.DataFrame(
            {
                'node_from': [f'N{a + 1}' for a, _ in pairs],
                'node_to': [f'N{b + 1}' for _, b in pairs],
                'capacity': [self.LINE_CAPACITY_MW + 100.0 * i for i in range(len(pairs))],
                'voltage': [380.0 if i % 2 == 0 else 220.0 for i in range(len(pairs))],
            },
            index=pd.Index([f'L{i + 1}' for i in range(len(pairs))], name='Line'),
        )

    def _line_pairs(self) -> list[tuple[int, int]]:
        n = self._scenario.n_nodes
        pairs = [(k - step, k) for k in range(1, n) for step in (1, 2) if k - step >= 0]
        if self._scenario.n_lines > len(pairs):
            raise ValueError(f'{n} nodes support at most {len(pairs)} lines.')
        return pairs[: self._scenario.n_lines]

    def _build_generators(self) -> pd.DataFrame:
        s = self._scenario
        max_generators = self.GENERATORS_PER_NODE * s.n_nodes
        if s.n_generators > max_generators:
            raise ValueError(f'{s.n_nodes} nodes support at most {max_generators} generators.')
        return pd.DataFrame(
            {
                'node': [f'N{i // self.GENERATORS_PER_NODE + 1}' for i in range(s.n_generators)],
                'fuel': [self.FUELS[i % len(self.FUELS)] for i in range(s.n_generators)],
                'capacity': [self.GENERATOR_CAPACITY_MW + 100.0 * i for i in range(s.n_generators)],
            },
            index=pd.Index([f'G{i + 1}' for i in range(s.n_generators)], name='Generator'),
        )

    def _build_demands(self) -> pd.DataFrame:
        s = self._scenario
        max_demands = self.DEMANDS_PER_NODE * s.n_nodes
        if s.n_demands > max_demands:
            raise ValueError(f'{s.n_nodes} nodes support at most {max_demands} demands.')
        return pd.DataFrame(
            {
                'node': [f'N{i // self.DEMANDS_PER_NODE + 1}' for i in range(s.n_demands)],
                'sector': [self.SECTORS[i % len(self.SECTORS)] for i in range(s.n_demands)],
                'base_demand': [500.0 + 50.0 * i for i in range(s.n_demands)],
            },
            index=pd.Index([f'D{i + 1}' for i in range(s.n_demands)], name='Demand'),
        )

    @staticmethod
    def _remove_objects(tables: dict[str, pd.DataFrame], removed: tuple[str, ...]) -> dict[str, pd.DataFrame]:
        removed = set(removed)
        lines = tables['lines']
        generators = tables['generators']
        demands = tables['demands']
        nodes = tables['nodes'].drop(index=removed, errors='ignore')
        nodes.loc[nodes['country'].isin(removed), 'country'] = None
        return {
            'countries': tables['countries'].drop(index=removed, errors='ignore'),
            'nodes': nodes,
            'lines': lines.loc[
                ~lines.index.isin(removed) & ~lines.node_from.isin(removed) & ~lines.node_to.isin(removed)
            ],
            'generators': generators.loc[~generators.index.isin(removed) & ~generators.node.isin(removed)],
            'demands': demands.loc[~demands.index.isin(removed) & ~demands.node.isin(removed)],
        }

    @staticmethod
    def _apply_text_changes(
        tables: dict[str, pd.DataFrame],
        changes: tuple[tuple[str, str, str, str], ...],
    ) -> dict[str, pd.DataFrame]:
        tables = {key: table.copy() for key, table in tables.items()}
        for table_name, object_id, column, value in changes:
            if object_id not in tables[table_name].index:
                raise KeyError(f'Object {object_id!r} not in table {table_name!r}.')
            tables[table_name].loc[object_id, column] = value
        return tables
