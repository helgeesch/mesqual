"""Raw tables and inventory of a synthetic scenario."""

from __future__ import annotations

import hashlib
import zlib
from dataclasses import dataclass
from functools import cached_property
from typing import ClassVar

import numpy as np
import pandas as pd

from mesqual.datasets.source_inventory import SourceInventory
from mesqual.testing.network import DummyNetwork
from mesqual.testing.scenario import DummyScenario


@dataclass(frozen=True)
class DummyInventory(SourceInventory):
    """What a synthetic source holds.

    Attributes:
        fingerprint: Hash of the scenario spec. Equal specs give equal fingerprints.
        variables: Names of the raw tables the source provides.
    """

    variables: tuple[str, ...]


class DummySource:
    """Raw tables of one synthetic scenario, as a simulation platform would write them.

    Model tables come from `DummyNetwork`. Time series tables have a time index
    named 'time' and one column per object. The values of an object on a day
    depend only on the seed, the date, the object id and its model properties.
    Adding or removing other objects leaves them unchanged.

    With `profile='daily'` the series look like those of a real power system:
    solar output is 0 at night and peaks around noon, longer and higher in
    summer; wind output varies slowly and is higher in winter; demand has a
    morning and an evening peak and is lower at night, on weekends and in
    summer; nuclear runs flat, coal and gas follow the net demand (demand minus
    solar and wind output); prices follow the net demand. All shapes are taken
    at the middle of each time step, and the noise is constant within an hour,
    so values at 15 minutes match those at 60 minutes. With `profile='constant'`
    every series is constant.

    Time series tables:

    - 'node_price': EUR/MWh per node.
    - 'generator_generation': MWh per period and generator.
    - 'demand_load': MW per demand, times `demand_factor`.
    - 'line_flow': MW per line, net flow where it enters the line, positive
      from node_from to node_to.
    - 'line_capacity': MW per line in line direction. The capacity against the
      line direction is `DOWN_CAPACITY_FACTOR` times this value.

    Args:
        scenario: The scenario spec.
    """

    MODEL_TABLES = ('countries', 'nodes', 'lines', 'generators', 'demands')
    TIME_SERIES_TABLES = ('node_price', 'generator_generation', 'demand_load', 'line_flow', 'line_capacity')
    TABLES = MODEL_TABLES + TIME_SERIES_TABLES
    LINE_LOSS_FACTOR = 0.02
    DOWN_CAPACITY_FACTOR = 0.9
    TIME_LEVEL = 'time'

    def __init__(self, scenario: DummyScenario):
        unknown = set(scenario.missing_tables) - set(self.TABLES)
        if unknown:
            raise ValueError(f'Unknown tables in missing_tables: {sorted(unknown)}. Use names of {self.TABLES}.')
        self.scenario = scenario
        self.network = DummyNetwork(scenario)
        self._tables: dict[str, pd.DataFrame] = {}

    @cached_property
    def inventory(self) -> DummyInventory:
        """Fingerprint and raw tables of this source."""
        fingerprint = hashlib.sha256(repr(self.scenario).encode()).hexdigest()[:16]
        variables = tuple(t for t in self.TABLES if t not in self.scenario.missing_tables)
        return DummyInventory(fingerprint=fingerprint, variables=variables)

    def has_table(self, table: str) -> bool:
        """True if the source provides the raw table."""
        return table in self.inventory.variables

    def read(self, table: str) -> pd.DataFrame:
        """Return a copy of a raw table.

        Raises:
            KeyError: If the source does not provide the table.
        """
        if not self.has_table(table):
            raise KeyError(f'Source of scenario {self.scenario.name!r} does not provide table {table!r}.')
        if table not in self._tables:
            self._tables[table] = self._build_table(table)
        return self._tables[table].copy()

    def _build_table(self, table: str) -> pd.DataFrame:
        if table in self.MODEL_TABLES:
            return self.network.tables[table]
        series = _SeriesBuilder(self.scenario, self.network).build(table)
        return self._apply_edge_cases(series)

    def _apply_edge_cases(self, df: pd.DataFrame) -> pd.DataFrame:
        s = self.scenario
        for start, end in s.nan_gaps:
            df.loc[self._window_mask(df.index, start, end)] = np.nan
        for start, end in s.missing_rows:
            df = df.loc[~self._window_mask(df.index, start, end)]
        if s.duplicate_rows:
            df = pd.concat([df, df.iloc[: s.duplicate_rows]])
        return self._convert_time_index(df)

    def _window_mask(self, index: pd.DatetimeIndex, start: str, end: str) -> np.ndarray:
        start, end = self.scenario.timestamp(start), self.scenario.timestamp(end)
        return (index >= start) & (index < end)

    def _convert_time_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Hook for sources with another time level. The default keeps the DatetimeIndex."""
        return df


class DummyPeriodSource(DummySource):
    """A `DummySource` with a time level of (run, period) tuples instead of timestamps.

    The run is the calendar day as an integer YYYYMMDD, for example 20240330, so
    the parts of a split scenario keep their run numbers. The period counts the
    time steps within the day, starting at 1.
    """

    def _convert_time_index(self, df: pd.DataFrame) -> pd.DataFrame:
        day_starts = df.index.normalize()
        local_days = day_starts.tz_localize(None) if day_starts.tz is not None else day_starts
        steps = pd.to_timedelta([self.scenario.resolution_on(day) for day in local_days])
        runs = local_days.strftime('%Y%m%d').astype(int)
        periods = ((df.index - day_starts) / steps).astype(int) + 1
        df = df.copy()
        df.index = pd.Index(list(zip(runs, periods)), tupleize_cols=False, name=self.TIME_LEVEL)
        return df


@dataclass(frozen=True)
class _DayGrid:
    """Time coordinates of the steps of one day, taken at the middle of each step.

    Attributes:
        index: Start of each step.
        day: The calendar day as a naive timestamp at midnight.
        step_hours: Length of a step in hours.
        clock_hours: Local clock time in hours.
        solar_hours: Local standard time in hours, without the DST offset.
        absolute_hours: Hours since `EPOCH` in UTC, or in naive time without time zone.
        hour_slots: Whole hours since local midnight, 0 to 24 on a fall-back day.
    """

    EPOCH: ClassVar[pd.Timestamp] = pd.Timestamp('2000-01-01')

    index: pd.DatetimeIndex
    day: pd.Timestamp
    step_hours: float
    clock_hours: np.ndarray
    solar_hours: np.ndarray
    absolute_hours: np.ndarray
    hour_slots: np.ndarray

    @classmethod
    def of(cls, scenario: DummyScenario, day: pd.Timestamp) -> _DayGrid:
        """Build the grid of one day of the scenario."""
        start = pd.Timestamp(day, tz=scenario.tz)
        end = pd.Timestamp(day + pd.Timedelta(days=1), tz=scenario.tz)
        step = pd.Timedelta(scenario.resolution_on(day))
        index = pd.date_range(start, end, freq=step, inclusive='left', name=DummySource.TIME_LEVEL)
        hour = pd.Timedelta(hours=1)
        half_step_hours = step / hour / 2
        clock_hours = (index.hour + index.minute / 60).to_numpy() + half_step_hours
        dst_hours = np.array([t.dst() / hour for t in index]) if scenario.tz else np.zeros(len(index))
        absolute = index.tz_convert('UTC').tz_localize(None) if scenario.tz else index
        return cls(
            index=index,
            day=day,
            step_hours=step / hour,
            clock_hours=clock_hours,
            solar_hours=clock_hours - dst_hours,
            absolute_hours=((absolute - cls.EPOCH) / hour).to_numpy() + half_step_hours,
            hour_slots=((index - start) // hour).to_numpy().astype(int),
        )

    @property
    def season_winter(self) -> float:
        """1 in mid January, -1 in mid July."""
        return float(np.cos(2 * np.pi * (self.day.day_of_year - 15) / 365))

    @property
    def season_summer(self) -> float:
        """1 at the June solstice, -1 at the December solstice."""
        return float(np.sin(2 * np.pi * (self.day.day_of_year - 80) / 365))

    @property
    def is_weekend(self) -> bool:
        """True on Saturday and Sunday."""
        return self.day.day_of_week >= 5


@dataclass(frozen=True)
class _SystemDay:
    """Shapes of one day that all objects of a scenario share, as fractions of their maximum.

    Attributes:
        demand: Demand, about 1 on a winter weekday evening.
        solar: Clear-sky solar output, 0 at night.
        wind: Wind output of the system.
    """

    SOLAR_SHARE: ClassVar[float] = 0.35
    WIND_SHARE: ClassVar[float] = 0.3
    NET_LOW: ClassVar[float] = 0.2
    NET_HIGH: ClassVar[float] = 1.1

    demand: np.ndarray
    solar: np.ndarray
    wind: np.ndarray

    @property
    def net_demand(self) -> np.ndarray:
        """Demand minus solar and wind output."""
        return self.demand - self.SOLAR_SHARE * self.solar - self.WIND_SHARE * self.wind

    @property
    def net_level(self) -> np.ndarray:
        """Net demand scaled to 0 (low) to 1 (high)."""
        return np.clip((self.net_demand - self.NET_LOW) / (self.NET_HIGH - self.NET_LOW), 0, 1)


class _Profiles:
    """Realistic shapes of demand, solar and wind output.

    All shapes depend only on the seed, the key parts and the time, so they are
    the same for a day in a split or unsplit scenario and at any resolution.

    Args:
        seed: Seed of the scenario.
        demand_factor: Factor on the demand of the system.
    """

    WIND_TERMS = 4
    WIND_PERIOD_HOURS = (30.0, 240.0)

    def __init__(self, seed: int, demand_factor: float = 1.0):
        self._seed = seed
        self._demand_factor = demand_factor

    def system_day(self, grid: _DayGrid) -> _SystemDay:
        """Return the shapes shared by all objects on the day of the grid, with the demand times the demand factor."""
        demand = self._demand_factor * self.demand(grid)
        return _SystemDay(demand=demand, solar=self.solar(grid), wind=self.wind(grid, 'system'))

    @classmethod
    def demand(cls, grid: _DayGrid) -> np.ndarray:
        """Demand with a morning and an evening peak, lower at night, on weekends and in summer."""
        h = grid.clock_hours
        daily = 0.72 + 0.18 * cls._bell(h, 8.5, 1.8) + 0.12 * cls._bell(h, 13.0, 3.0) + 0.25 * cls._bell(h, 19.0, 2.0)
        weekly = 0.85 if grid.is_weekend else 1.0
        return daily * weekly * (1 + 0.12 * grid.season_winter)

    @staticmethod
    def solar(grid: _DayGrid) -> np.ndarray:
        """Clear-sky solar output: a bell around noon in standard time, longer and higher in summer."""
        day_length = 12 + 4.5 * grid.season_summer
        peak = 0.5 + 0.3 * grid.season_summer
        x = (grid.solar_hours - 12) / day_length
        return np.where(np.abs(x) < 0.5, peak * np.cos(np.pi * x), 0.0)

    def wind(self, grid: _DayGrid, *key: str) -> np.ndarray:
        """Slow wind output without a daily cycle, higher in winter.

        A sum of sine waves with periods of 30 to 240 hours. Their periods,
        phases and weights come from the seed and the key.
        """
        rng = np.random.default_rng([self._seed, *(zlib.crc32(k.encode()) for k in ('wind', *key))])
        periods = rng.uniform(*self.WIND_PERIOD_HOURS, self.WIND_TERMS)
        phases = rng.uniform(0, 2 * np.pi, self.WIND_TERMS)
        weights = rng.uniform(0.5, 1.0, self.WIND_TERMS)
        waves = np.sin(2 * np.pi * grid.absolute_hours[None, :] / periods[:, None] + phases[:, None])
        wave = weights @ waves / weights.sum()
        level = 0.35 + 0.15 * grid.season_winter
        return np.clip(level * (1 + 1.5 * wave), 0, 1)

    @staticmethod
    def _bell(hours: np.ndarray, center: float, width: float) -> np.ndarray:
        return np.exp(-0.5 * ((hours - center) / width) ** 2)


class _SeriesBuilder:
    """Builds the time series tables of a synthetic scenario day by day.

    With `profile='daily'`:

    - Demand has a morning and an evening peak, is lower at night, on weekends
      and in summer.
    - Solar output is 0 at night and a bell around noon, longer and higher in
      summer, times a cloud factor per generator and day.
    - Wind output varies slowly, without a daily cycle, and is higher in winter.
    - Nuclear runs high and flat. Coal and gas, and any other fuel, follow the
      net demand (demand minus solar and wind output) within their capacity.
    - Prices follow the net demand. A higher `demand_factor` raises the net
      demand, so it also raises prices and the output of coal and gas.
    - Line flows follow the net demand in a fixed direction per line plus a
      slow random part, and stay below the line capacity. The line capacity is
      higher in winter.

    With `profile='constant'` every series is constant.

    Args:
        scenario: The scenario spec.
        network: The network of the scenario.
    """

    NOISE_SCALE: ClassVar[dict[str, float]] = {
        'node_price': 3.0,
        'demand_load': 0.03,
        'generator_generation': 0.03,
        'line_flow': 0.05,
    }
    FLOW_LIMITS = (-0.8, 0.85)

    def __init__(self, scenario: DummyScenario, network: DummyNetwork):
        self._scenario = scenario
        self._network = network
        self._profiles = _Profiles(scenario.seed, scenario.demand_factor)

    def build(self, table: str) -> pd.DataFrame:
        """Return the time series table for the whole scenario horizon."""
        objects = self._objects_of(table)
        days = [self._build_day(table, objects, day) for day in self._scenario.day_list]
        return pd.concat(days)

    def _objects_of(self, table: str) -> pd.DataFrame:
        if table == 'node_price':
            return self._network.nodes
        if table == 'generator_generation':
            return self._network.generators
        if table == 'demand_load':
            return self._network.demands
        if table in ('line_flow', 'line_capacity'):
            return self._network.lines
        raise ValueError(f'Unknown time series table {table!r}.')

    def _build_day(self, table: str, objects: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:
        grid = _DayGrid.of(self._scenario, day)
        if self._scenario.profile == 'constant':
            columns = {
                obj: self._constant(table, objects.loc[obj], grid) * np.ones(len(grid.index)) for obj in objects.index
            }
        else:
            system = self._profiles.system_day(grid)
            columns = {obj: self._daily(table, objects.loc[obj], grid, system) for obj in objects.index}
        return pd.DataFrame(columns, index=grid.index, columns=objects.index, dtype=float)

    def _constant(self, table: str, obj: pd.Series, grid: _DayGrid) -> float:
        if table == 'node_price':
            return obj.base_price
        if table == 'demand_load':
            return self._scenario.demand_factor * obj.base_demand
        if table == 'generator_generation':
            return 0.5 * obj.capacity * grid.step_hours
        if table == 'line_flow':
            return 0.5 * obj.capacity
        if table == 'line_capacity':
            return obj.capacity
        raise ValueError(f'Unknown time series table {table!r}.')

    def _daily(self, table: str, obj: pd.Series, grid: _DayGrid, system: _SystemDay) -> np.ndarray:
        rng = self._rng(table, obj, grid.day)
        noise = self.NOISE_SCALE.get(table, 0.0) * rng.standard_normal(25)[grid.hour_slots]
        if table == 'node_price':
            return obj.base_price * (0.3 + 1.4 * system.net_level) + noise
        if table == 'demand_load':
            return obj.base_demand * system.demand * (1 + noise)
        if table == 'generator_generation':
            utilisation = self._utilisation(obj, grid, system, noise, rng)
            return obj.capacity * np.clip(utilisation, 0, 1) * grid.step_hours
        if table == 'line_flow':
            direction = 1 if zlib.crc32(str(obj.name).encode()) % 2 == 0 else -1
            own = self._profiles.wind(grid, 'line', str(obj.name))
            loading = direction * (0.15 + 0.5 * system.net_level) + 0.3 * (own - 0.35) + noise
            return obj.capacity * np.clip(loading, *self.FLOW_LIMITS)
        if table == 'line_capacity':
            return obj.capacity * (0.95 + 0.05 * grid.season_winter) * np.ones(len(grid.index))
        raise ValueError(f'Unknown time series table {table!r}.')

    def _utilisation(
        self,
        generator: pd.Series,
        grid: _DayGrid,
        system: _SystemDay,
        noise: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        if generator.fuel == 'nuclear':
            return 0.92 + noise
        if generator.fuel == 'solar':
            cloud_factor = rng.uniform(0.3, 1.0)
            return system.solar * cloud_factor * (1 + 5 * noise)
        if generator.fuel == 'wind':
            return 0.6 * system.wind + 0.4 * self._profiles.wind(grid, 'generator', str(generator.name)) + noise
        if generator.fuel == 'coal':
            return 0.35 + 0.55 * system.net_level + noise
        return 0.05 + 0.9 * system.net_level + noise

    def _rng(self, table: str, obj: pd.Series, day: pd.Timestamp) -> np.random.Generator:
        key = [self._scenario.seed, zlib.crc32(table.encode()), zlib.crc32(str(obj.name).encode()), day.toordinal()]
        return np.random.default_rng(key)
