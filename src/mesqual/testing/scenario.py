"""Scenario spec of the synthetic data generator."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class DummyScenario:
    """Spec of one synthetic scenario.

    The generator derives the network, the model tables and the time series from
    these fields. Equal specs give equal data. The values of an object on a day
    depend only on the seed, the date, the object id and its model properties. So
    a scenario split into parts gives the same values as the unsplit scenario,
    and more objects (for example a higher `n_nodes`) or removed objects leave
    the other objects unchanged.

    Attributes:
        name: Scenario name. It becomes the dataset name.
        start: First day, for example '2024-03-30'.
        days: Number of days.
        tz: Time zone of the time index, or None for naive timestamps.
        resolution: Time step at the start, for example '60min'.
        resolution_change: Optional (date, resolution): from that date on the
            series use the new time step, for example ('2024-03-31', '15min').
        seed: Seed of the random noise.
        profile: 'daily' for realistic daily, weekly and seasonal profiles with
            noise (see `DummySource`), 'constant' for constant values, so that
            tests can compute expected values by hand.
        demand_factor: Factor on the demand of every demand object.
        n_countries: Number of countries, at least 1. Nodes are placed round
            robin in the countries, so a change of `n_countries` changes the
            country and the location of existing nodes.
        n_nodes: Number of nodes, placed round robin in the countries.
        n_lines: Number of lines. Line k connects the nodes in a fixed sequence:
            each new node connects to the previous node and to the node before that.
        n_generators: Number of generators. Generators k and k+1 (k even) sit at
            node k/2+1, so at most two generators per node. The fuels repeat in
            the order of `DummyNetwork.FUELS`; the default gives one generator per fuel.
        n_demands: Number of demands, placed like the generators: demands k and
            k+1 (k even) sit at node k/2+1. The default gives every node two demands.
        node_without_country: If True, adds the node 'N0' without country, lines
            and demands.
        removed_objects: Object ids to remove. A removed node also removes its
            lines, generators and demands. The nodes of a removed country keep
            their place with no country (`country` is None).
        text_changes: (model table, object, column, value) entries that change
            text properties, for example ('generators', 'G1', 'fuel', 'gas').
        missing_tables: Raw tables the source does not provide, for example
            ('demand_load',). See `DummySource.TABLES`.
        nan_gaps: (start, end) windows where all time series values are NaN.
            The end is exclusive. A timestamp without offset is local time in
            `tz`; one with offset, for example '2024-10-27 02:00+01:00', selects
            one of the two hours of a DST fall-back day.
        missing_rows: (start, end) windows where the time series have no rows.
        duplicate_rows: Number of rows at the start of each time series that
            appear twice. The copies are appended at the end, so the raw index
            is not monotonic.
    """

    name: str = 'base'
    start: str = '2024-01-01'
    days: int = 2
    tz: str | None = 'Europe/Berlin'
    resolution: str = '60min'
    resolution_change: tuple[str, str] | None = None
    seed: int = 0
    profile: str = 'daily'
    demand_factor: float = 1.0
    n_countries: int = 2
    n_nodes: int = 4
    n_lines: int = 5
    n_generators: int = 5
    n_demands: int = 8
    node_without_country: bool = False
    removed_objects: tuple[str, ...] = ()
    text_changes: tuple[tuple[str, str, str, str], ...] = ()
    missing_tables: tuple[str, ...] = ()
    nan_gaps: tuple[tuple[str, str], ...] = ()
    missing_rows: tuple[tuple[str, str], ...] = ()
    duplicate_rows: int = 0

    PROFILES = ('daily', 'constant')

    def __post_init__(self):
        if self.days < 1:
            raise ValueError(f'days must be at least 1, got {self.days}.')
        if self.profile not in self.PROFILES:
            raise ValueError(f'Unknown profile {self.profile!r}. Use one of {self.PROFILES}.')
        if self.n_countries < 1:
            raise ValueError(f'n_countries must be at least 1, got {self.n_countries}.')
        if self.n_nodes < self.n_countries:
            raise ValueError('n_nodes must be at least n_countries.')

    @property
    def first_day(self) -> pd.Timestamp:
        """First day as a naive timestamp at midnight."""
        return pd.Timestamp(self.start).normalize()

    @property
    def last_day(self) -> pd.Timestamp:
        """Last day as a naive timestamp at midnight."""
        return self.first_day + pd.Timedelta(days=self.days - 1)

    @property
    def day_list(self) -> list[pd.Timestamp]:
        """All days of the scenario as naive timestamps at midnight."""
        return list(pd.date_range(self.first_day, self.last_day, freq='D'))

    def resolution_on(self, day: pd.Timestamp) -> str:
        """Time step of the series on the given day."""
        if self.resolution_change is not None:
            change_day, new_resolution = self.resolution_change
            if day >= pd.Timestamp(change_day):
                return new_resolution
        return self.resolution

    def timestamp(self, value: str) -> pd.Timestamp:
        """Parse a timestamp string in the time zone of the scenario.

        A string without offset is local time in `tz`. A string with offset is
        converted to `tz`.
        """
        parsed = pd.Timestamp(value)
        if parsed.tzinfo is None:
            return parsed.tz_localize(self.tz) if self.tz else parsed
        if self.tz is None:
            raise ValueError(f'Timestamp {value!r} has an offset, but the scenario has no time zone.')
        return parsed.tz_convert(self.tz)

    def replace(self, **changes) -> DummyScenario:
        """Return a copy with the given fields changed."""
        return dataclasses.replace(self, **changes)

    def split(self, at: str) -> tuple[DummyScenario, DummyScenario]:
        """Split the scenario into two parts with consecutive date ranges.

        Args:
            at: First day of the second part. It must lie after the first day
                and not after the last day.

        Returns:
            The parts '<name>.1' and '<name>.2'. Together they give the same data
                as the unsplit scenario.
        """
        split_day = pd.Timestamp(at).normalize()
        days_first = (split_day - self.first_day).days
        if not 0 < days_first < self.days:
            raise ValueError(f'Split day {at} must lie after {self.start} and within the {self.days} days.')
        first = self.replace(name=f'{self.name}.1', days=days_first)
        second = self.replace(name=f'{self.name}.2', start=split_day.strftime('%Y-%m-%d'), days=self.days - days_first)
        return first, second
