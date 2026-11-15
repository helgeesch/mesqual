"""Synthetic platform dataset with its interpreters."""

from __future__ import annotations

import re
import string
from dataclasses import dataclass
from typing import ClassVar

import pandas as pd

from mesqual.cache import Cache
from mesqual.datasets.dataset import Dataset
from mesqual.datasets.dataset_collection import DatasetMergeCollection
from mesqual.datasets.dataset_config import DatasetConfig, config_field
from mesqual.datasets.describe import Availability
from mesqual.datasets.platform_dataset import PlatformDataset
from mesqual.enums import TopologyTypeEnum
from mesqual.flag import (
    AreaFlagSpec,
    AreaLayer,
    BorderFlagSpec,
    BorderLayer,
    Dimension,
    FlagIndex,
    ModelFlagSpec,
    TimeSeriesFlagSpec,
    VariantValues,
)
from mesqual.layer_flags import (
    AreaFlagInterpreter,
    BorderCapacitySum,
    BorderFlagInterpreter,
    BorderFlowSum,
    BorderModel,
    PriceSpread,
    Sum,
    WeightedMean,
)
from mesqual.testing.scenario import DummyScenario
from mesqual.testing.source import DummyInventory, DummyPeriodSource, DummySource
from mesqual.units import Units


@dataclass
class DummyConfig(DatasetConfig):
    """Config of `DummyDataset`.

    Attributes:
        price_offset: Value in EUR/MWh added to every node price. Default: 0.0.
    """

    price_offset: float = config_field(0.0, 'Value in EUR/MWh added to every node price.')


class DummyInterpreter(Dataset[DummyConfig, str, FlagIndex]):
    """Base of the interpreters of `DummyDataset`.

    A subclass declares its flags in `flag_specs`. A spec that reads a raw
    table names it in `SPEC_TABLE`. A derived spec lists the flags it
    fetches through the parent dataset in `INPUT_FLAGS`. An input flag may
    use the variant placeholders of the spec, for example
    `'Demand.Results.demand_{product}'`. `_is_available` reads one of the
    two, so every spec must be in exactly one. A subclass that breaks this,
    or whose input flag has a placeholder that is no variant of its spec,
    raises `ValueError` at class creation.

    Args:
        source: The raw tables of the scenario.
        parent_dataset: The platform dataset that holds this interpreter.
    """

    SPEC_TABLE: ClassVar[dict[str, str]] = {}
    INPUT_FLAGS: ClassVar[dict[str, tuple[str, ...]]] = {}

    def __init__(self, source: DummySource, parent_dataset: Dataset | None = None):
        super().__init__(parent_dataset=parent_dataset)
        self.source = source

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        both = set(cls.SPEC_TABLE) & set(cls.INPUT_FLAGS)
        if both:
            raise ValueError(f'{cls.__name__}: specs {sorted(both)} are in SPEC_TABLE and in INPUT_FLAGS.')
        declared = cls.SPEC_TABLE.keys() | cls.INPUT_FLAGS.keys()
        missing = [spec.name for spec in cls.flag_specs if spec.name not in declared]
        if missing:
            raise ValueError(f'{cls.__name__}: specs {missing} are in neither SPEC_TABLE nor INPUT_FLAGS.')
        for spec in cls.flag_specs:
            for input_flag in cls.INPUT_FLAGS.get(spec.name, ()):
                fields = {name for _, name, _, _ in string.Formatter().parse(input_flag) if name is not None}
                unknown = fields - {dimension.name for dimension in spec.variants}
                if unknown:
                    raise ValueError(
                        f'{cls.__name__}: input flag {input_flag!r} of {spec.name!r} has placeholders '
                        f'{sorted(unknown)} that are no variants of the spec.'
                    )

    @classmethod
    def get_config_type(cls) -> type[DummyConfig]:
        """Return `DummyConfig`."""
        return DummyConfig

    def _is_available(self, flag: str) -> Availability:
        """Availability from the raw table in the source inventory, or from the input flags.

        A spec in `SPEC_TABLE` is available if the source inventory lists its
        table; the reason names the table. For a spec in `INPUT_FLAGS`,
        `Availability.of_inputs` combines the availability of the input flags.
        """
        spec_name = self.flag_index.get_spec(flag).name
        if spec_name in self.SPEC_TABLE:
            table = self.SPEC_TABLE[spec_name]
            has_table = self.source.has_table(table)
            return Availability(has_table, f'table {table!r} {"in" if has_table else "not in"} the source')
        return Availability.of_inputs({f: self.parent_dataset.availability(f) for f in self._input_flags(flag)})

    def _input_flags(self, flag: str) -> list[str]:
        """The `INPUT_FLAGS` of the spec of `flag`, with the variant values of `flag` filled in."""
        spec = self.flag_index.get_spec(flag)
        variant = spec.variant_of(flag).as_dict()
        return [input_flag.format(**variant) for input_flag in self.INPUT_FLAGS[spec.name]]

    def _fetch_inputs(self, flag: str, config: DummyConfig, params: dict) -> list[pd.DataFrame]:
        """Fetch the `INPUT_FLAGS` of the spec of `flag` through the parent dataset, so overrides reach them.

        Each input gets the parameters of `params` that its spec declares.
        """
        inputs = []
        for input_flag in self._input_flags(flag):
            declared = {dimension.name for dimension in self.flag_index.get_spec(input_flag).params}
            inputs.append(
                self.parent_dataset.fetch(input_flag, config, **{k: v for k, v in params.items() if k in declared})
            )
        return inputs


class DummyDataset(PlatformDataset[DummyInterpreter, DummyConfig, str, FlagIndex]):
    """Platform dataset of one synthetic scenario.

    Flags: the model tables 'Country.Model', 'Node.Model', 'Line.Model',
    'Generator.Model' and 'Demand.Model'; time series of node prices,
    demand, generation, line flows and line capacities; the derived line
    loading, node demand and generation per node and fuel; the area layer
    'Country' on 'Node.Model' and the border layer 'CountryBorder' on
    'Line.Model'. `describe_class_flags()` lists each spec with its unit,
    variants, parameters and description.

    A flag whose raw table the scenario lists in `missing_tables` is
    unavailable, and so is each flag that reads it. `fetch` raises
    `FlagUnavailableError`, a `ValueError`, for it. Without the table
    'countries', the geometry columns of the border model are NaN.

    Default attributes: 'first_date', 'last_date' and 'days' from the scenario.
    Explicit attributes win.

    Args:
        scenario: The scenario spec.
        name: Dataset name. Default: the scenario name.
        attributes: Extra attributes.
        cache: Optional cache of fetch results.
        config: Optional instance config.

    Examples:
        >>> from mesqual.testing import DummyScenario
        >>> dataset = DummyDataset(DummyScenario(name='base', days=3))
        >>> prices = dataset.fetch('Node.Results.price')
    """

    SOURCE_TYPE: ClassVar[type[DummySource]] = DummySource

    def __init__(
        self,
        scenario: DummyScenario,
        name: str | None = None,
        attributes: dict | None = None,
        cache: Cache | None = None,
        config: DummyConfig | None = None,
    ):
        self.scenario = scenario
        self.source = self.SOURCE_TYPE(scenario)
        super().__init__(
            name=name or scenario.name,
            attributes=attributes,
            cache=cache,
            config=config,
            source=self.source,
        )

    def build_source_inventory(self) -> DummyInventory:
        """Return the inventory of the synthetic source: its fingerprint and raw tables."""
        return self.source.inventory

    def _default_attributes(self) -> dict:
        return {
            'first_date': self.scenario.first_day.strftime('%Y-%m-%d'),
            'last_date': self.scenario.last_day.strftime('%Y-%m-%d'),
            'days': self.scenario.days,
        }

    @classmethod
    def get_config_type(cls) -> type[DummyConfig]:
        """Return `DummyConfig`."""
        return DummyConfig

    @classmethod
    def get_child_dataset_type(cls) -> type[DummyInterpreter]:
        """Return `DummyInterpreter`, the base of all interpreters of this dataset."""
        return DummyInterpreter


@DummyDataset.register_interpreter
class DummyModelInterpreter(DummyInterpreter):
    """Model tables of countries, nodes, lines, generators and demands."""

    LINE_MODEL_SPEC = ModelFlagSpec(
        name='Line.Model',
        description='Lines between two nodes. The line direction runs from node_from to node_to.',
        index='Line',
        topology_type=TopologyTypeEnum.Edge,
        memberships={'node_from': 'Node.Model', 'node_to': 'Node.Model'},
        edge_ends=('node_from', 'node_to'),
    )
    flag_specs = (
        ModelFlagSpec(
            name='Country.Model',
            description='Countries, each with a square polygon.',
            index='Country',
            topology_type=TopologyTypeEnum.Area,
        ),
        ModelFlagSpec(
            name='Node.Model',
            description='Nodes of the network, each with a point and a country.',
            index='Node',
            topology_type=TopologyTypeEnum.Node,
            memberships={'country': 'Country.Model'},
        ),
        LINE_MODEL_SPEC,
        ModelFlagSpec(
            name='Generator.Model',
            description='Generators, each at one node and with one fuel.',
            index='Generator',
            topology_type=TopologyTypeEnum.NodeConnectedElement,
            memberships={'node': 'Node.Model'},
        ),
        ModelFlagSpec(
            name='Demand.Model',
            description='Demands, each at one node and with one sector.',
            index='Demand',
            topology_type=TopologyTypeEnum.NodeConnectedElement,
            memberships={'node': 'Node.Model'},
        ),
    )
    SPEC_TABLE: ClassVar[dict[str, str]] = {
        'Country.Model': 'countries',
        'Node.Model': 'nodes',
        'Line.Model': 'lines',
        'Generator.Model': 'generators',
        'Demand.Model': 'demands',
    }

    def _fetch(self, flag: str, effective_config: DummyConfig, **params) -> pd.DataFrame:
        return self.source.read(self.SPEC_TABLE[flag])


def _product_values(shares: dict[str, float]) -> dict[str, str]:
    """The values of the `product` variant, each with its share of the total demand."""
    orders = {'total': 'all orders', 'hourly': 'hourly orders only', 'quarter_hourly': 'quarter-hourly orders only'}
    return {
        product: orders[product] if product == 'total' else f'{orders[product]}, {share:.0%} of the total'
        for product, share in shares.items()
    }


SOLUTION = Dimension(
    'solution',
    'Solution of the run to read.',
    {
        'first': 'the first solution of the run',
        'final': 'the final solution of the run',
    },
    default='final',
)
"""Parameter of the node price and the demand. The first solution deviates from the final one by a fixed amount."""


@DummyDataset.register_interpreter
class DummyTimeSeriesInterpreter(DummyInterpreter):
    """Node prices, generation, demand, line flows and line capacities, each from one raw table."""

    PRODUCT_SHARES: ClassVar[dict[str, float]] = {'total': 1.0, 'hourly': 0.75, 'quarter_hourly': 0.25}
    SOLUTION_PRICE_STEP: ClassVar[dict[str, float]] = {'first': -1.0, 'final': 0.0}
    """EUR/MWh added to the final node price per solution, times the number in the node id (N3: 3)."""
    SOLUTION_DEMAND_FACTOR: ClassVar[dict[str, float]] = {'first': 0.98, 'final': 1.0}
    """Factor on the final total demand, per solution."""
    PRICE_SPEC = TimeSeriesFlagSpec(
        name='Node.Results.price',
        description='Price at the node.',
        model_flag='Node.Model',
        unit=Units.EUR_per_MWh,
        params=(SOLUTION,),
    )
    GENERATION_SPEC = TimeSeriesFlagSpec(
        name='Generator.Results.generation',
        description='Energy generated in each period.',
        model_flag='Generator.Model',
        unit=Units.MWh,
        fill_value=0,
    )
    DEMAND_SPEC = TimeSeriesFlagSpec(
        name='Demand.Results.demand_{product}',
        description='Demand of the demand object.',
        model_flag='Demand.Model',
        unit=Units.MW,
        fill_value=0,
        variants=(Dimension('product', 'Order products to include.', _product_values(PRODUCT_SHARES)),),
        params=(SOLUTION,),
    )
    CAPACITY_SPEC = TimeSeriesFlagSpec(
        name='Line.Constraints.capacity_{direction}',
        description='Transfer capacity of the line.',
        model_flag='Line.Model',
        unit=Units.MW,
        fill_value=0,
        variants=(
            Dimension(
                'direction',
                'Direction of the capacity.',
                {
                    'up': 'from node_from to node_to',
                    'down': 'from node_to to node_from, 90% of up',
                },
            ),
        ),
    )
    FLOW_SPEC = TimeSeriesFlagSpec(
        name='Line.Results.flow_{side}_{direction}',
        description='Power flow on the line.',
        model_flag='Line.Model',
        unit=Units.MW,
        fill_value=0,
        variants=(
            Dimension(
                'side',
                'Where the flow is measured.',
                {
                    'sent': 'where the flow enters the line, before losses',
                    'received': 'where the flow leaves the line, after losses',
                },
            ),
            Dimension(
                'direction',
                'Flow direction.',
                {
                    'up': 'from node_from to node_to, zero otherwise',
                    'down': 'from node_to to node_from, zero otherwise',
                    'net': 'up minus down',
                },
            ),
        ),
    )
    flag_specs = (PRICE_SPEC, GENERATION_SPEC, DEMAND_SPEC, CAPACITY_SPEC, FLOW_SPEC)
    SPEC_TABLE: ClassVar[dict[str, str]] = {
        PRICE_SPEC.name: 'node_price',
        GENERATION_SPEC.name: 'generator_generation',
        DEMAND_SPEC.name: 'demand_load',
        CAPACITY_SPEC.name: 'line_capacity',
        FLOW_SPEC.name: 'line_flow',
    }

    def _fetch(self, flag: str, effective_config: DummyConfig, **params) -> pd.DataFrame:
        spec = self.flag_index.get_spec(flag)
        variant = spec.variant_of(flag)
        results = {
            self.PRICE_SPEC.name: lambda: self._price(effective_config, params['solution']),
            self.GENERATION_SPEC.name: lambda: self._table(spec),
            self.DEMAND_SPEC.name: lambda: self._demand(variant, effective_config, params['solution']),
            self.CAPACITY_SPEC.name: lambda: self._line_capacity(variant, effective_config),
            self.FLOW_SPEC.name: lambda: self._line_flow(variant, effective_config),
        }
        return results[spec.name]()

    def _table(self, spec: TimeSeriesFlagSpec) -> pd.DataFrame:
        return self.source.read(self.SPEC_TABLE[spec.name])

    def _price(self, config: DummyConfig, solution: str) -> pd.DataFrame:
        prices = self._table(self.PRICE_SPEC) + config.price_offset
        node_numbers = [int(re.search(r'\d+$', node).group()) for node in prices.columns]
        return prices + self.SOLUTION_PRICE_STEP[solution] * pd.Series(node_numbers, index=prices.columns, dtype=float)

    def _demand(self, variant: VariantValues, config: DummyConfig, solution: str) -> pd.DataFrame:
        """'total' is the raw table. 'hourly' and 'quarter_hourly' are shares of 'total' through the parent dataset."""

        def share() -> pd.DataFrame:
            total = self._variant(self.DEMAND_SPEC, config, {'product': 'total'}, solution=solution)
            return total * self.PRODUCT_SHARES[variant.product]

        products = {
            'total': lambda: self._table(self.DEMAND_SPEC) * self.SOLUTION_DEMAND_FACTOR[solution],
            'hourly': share,
            'quarter_hourly': share,
        }
        return products[variant.product]()

    def _line_capacity(self, variant: VariantValues, config: DummyConfig) -> pd.DataFrame:
        """'up' is the raw table, 'down' derives from 'up' through the parent dataset."""
        capacities = {
            'up': lambda: self._table(self.CAPACITY_SPEC),
            'down': lambda: (
                self._variant(self.CAPACITY_SPEC, config, {'direction': 'up'}) * self.source.DOWN_CAPACITY_FACTOR
            ),
        }
        return capacities[variant.direction]()

    def _line_flow(self, variant: VariantValues, config: DummyConfig) -> pd.DataFrame:
        """'sent' 'net' is the raw table. The other variants derive from it through the parent dataset.

        'sent' 'up' and 'down' are the positive and the negative part of
        'sent' 'net'. A 'received' variant is the 'sent' variant minus the
        line losses.
        """
        sent_flows = {
            'net': lambda: self._table(self.FLOW_SPEC),
            'up': lambda: self._sent_flow('net', config).clip(lower=0),
            'down': lambda: (-self._sent_flow('net', config)).clip(lower=0),
        }
        sides = {
            'sent': lambda: sent_flows[variant.direction](),
            'received': lambda: self._sent_flow(variant.direction, config) * (1.0 - self.source.LINE_LOSS_FACTOR),
        }
        return sides[variant.side]()

    def _sent_flow(self, direction: str, config: DummyConfig) -> pd.DataFrame:
        return self._variant(self.FLOW_SPEC, config, {'side': 'sent', 'direction': direction})

    def _variant(
        self,
        spec: TimeSeriesFlagSpec,
        config: DummyConfig,
        variant: dict[str, str],
        **params,
    ) -> pd.DataFrame:
        """Fetch another variant of `spec` through the parent dataset, so an override of it reaches this one."""
        return self.parent_dataset.fetch(spec.name.format(**variant), config, **params)


@DummyDataset.register_interpreter
class DummyDerivedInterpreter(DummyInterpreter):
    """Line loading, node demand and generation per node and fuel, from their `INPUT_FLAGS`.

    Nodes without demands or generators have no column.
    """

    LOADING_SPEC = TimeSeriesFlagSpec(
        name='Line.Results.loading',
        description='Absolute net flow where it enters the line, divided by the capacity in flow direction.',
        model_flag='Line.Model',
        unit=Units.per_unit,
    )
    NODE_DEMAND_SPEC = TimeSeriesFlagSpec(
        name='Node.Results.demand_{product}',
        description='Sum of the demand of the demands at the node.',
        model_flag='Node.Model',
        unit=Units.MW,
        fill_value=0,
        variants=DummyTimeSeriesInterpreter.DEMAND_SPEC.variants,
        params=DummyTimeSeriesInterpreter.DEMAND_SPEC.params,
    )
    GENERATION_BY_FUEL_SPEC = TimeSeriesFlagSpec(
        name='Node.Results.generation_{fuel}',
        description='Generation per node.',
        model_flag='Node.Model',
        unit=Units.MWh,
        fill_value=0,
        variants=(Dimension('fuel', 'Fuel of the generators.', lambda ds: DummyDerivedInterpreter._fuels(ds)),),
    )
    flag_specs = (LOADING_SPEC, NODE_DEMAND_SPEC, GENERATION_BY_FUEL_SPEC)
    INPUT_FLAGS: ClassVar[dict[str, tuple[str, ...]]] = {
        LOADING_SPEC.name: (
            'Line.Results.flow_sent_net',
            'Line.Constraints.capacity_up',
            'Line.Constraints.capacity_down',
        ),
        NODE_DEMAND_SPEC.name: ('Demand.Model', 'Demand.Results.demand_{product}'),
        GENERATION_BY_FUEL_SPEC.name: ('Generator.Model', 'Generator.Results.generation'),
    }

    def _fetch(self, flag: str, effective_config: DummyConfig, **params) -> pd.DataFrame:
        inputs = self._fetch_inputs(flag, effective_config, params)
        results = {
            self.LOADING_SPEC.name: lambda: self._loading(*inputs),
            self.NODE_DEMAND_SPEC.name: lambda: self._demand_per_node(*inputs),
            self.GENERATION_BY_FUEL_SPEC.name: lambda: self._generation_per_node(flag, *inputs),
        }
        return results[self.flag_index.get_spec(flag).name]()

    @staticmethod
    def _fuels(dataset: Dataset) -> dict[str, str]:
        """The fuels of the generator model."""
        return {f: f'from {f} generators' for f in sorted(dataset.fetch('Generator.Model')['fuel'].unique())}

    @staticmethod
    def _loading(flow: pd.DataFrame, capacity_up: pd.DataFrame, capacity_down: pd.DataFrame) -> pd.DataFrame:
        capacity = capacity_up.where(flow >= 0, capacity_down)
        return flow.abs() / capacity

    def _demand_per_node(self, demands: pd.DataFrame, demand: pd.DataFrame) -> pd.DataFrame:
        return self._sum_per_node(demand, demands['node'])

    def _generation_per_node(self, flag: str, generators: pd.DataFrame, generation: pd.DataFrame) -> pd.DataFrame:
        ids = generators.index[generators['fuel'] == self.GENERATION_BY_FUEL_SPEC.variant_of(flag).fuel]
        return self._sum_per_node(generation[ids], generators.loc[ids, 'node'])

    @staticmethod
    def _sum_per_node(series: pd.DataFrame, node_of_column: pd.Series) -> pd.DataFrame:
        per_node = series.T.groupby(node_of_column).sum(min_count=1).T
        return per_node.rename_axis(columns='Node')


DummyDataset.register_area_layer(
    AreaLayer(
        name='Country',
        index='Country',
        base_model='Node.Model',
        layer_model='Country.Model',
        description='Countries, each a group of nodes through the node column country.',
    )
)
DummyDataset.register_border_layer(
    BorderLayer(
        name='CountryBorder',
        index='CountryBorder',
        area_layer='Country',
        edge_model='Line.Model',
        description='Borders between two countries, crossed by lines.',
    )
)


@DummyDataset.register_interpreter
class DummyAreaFlags(AreaFlagInterpreter):
    """Flags on the area layer 'Country'.

    The layer model is the declared 'Country.Model', so there is no area model spec.
    """

    PRICE_SPEC = AreaFlagSpec(
        name='{area_layer}.Results.price',
        base=DummyTimeSeriesInterpreter.PRICE_SPEC,
        aggregation=WeightedMean(weight='Node.Results.demand_total'),
        description='Mean of the node prices of the area, weighted by the total node demand.',
    )
    DEMAND_SPEC = AreaFlagSpec(
        name='{area_layer}.Results.demand_{product}',
        base=DummyDerivedInterpreter.NODE_DEMAND_SPEC,
        aggregation=Sum(),
        description='Sum of the node demand of the area.',
    )
    GENERATION_BY_FUEL_SPEC = AreaFlagSpec(
        name='{area_layer}.Results.generation_{fuel}',
        base=DummyDerivedInterpreter.GENERATION_BY_FUEL_SPEC,
        aggregation=Sum(),
        description='Sum of the node generation of the area.',
    )
    flag_specs = (PRICE_SPEC, DEMAND_SPEC, GENERATION_BY_FUEL_SPEC)


@DummyDataset.register_interpreter
class DummyBorderFlags(BorderFlagInterpreter):
    """Flags on the border layer 'CountryBorder'."""

    MODEL_SPEC = BorderFlagSpec(
        name='{border_layer}.Model',
        base=DummyModelInterpreter.LINE_MODEL_SPEC,
        aggregation=BorderModel(),
        description='One row per border: area_from, area_to, the lines in and against border direction, geometry.',
    )
    FLOW_SPEC = BorderFlagSpec(
        name='{border_layer}.Results.flow_{side}_{direction}',
        base=DummyTimeSeriesInterpreter.FLOW_SPEC,
        aggregation=BorderFlowSum(),
        description='Sum of the line flows across the border, in border direction.',
    )
    CAPACITY_SPEC = BorderFlagSpec(
        name='{border_layer}.Constraints.capacity_{direction}',
        base=DummyTimeSeriesInterpreter.CAPACITY_SPEC,
        aggregation=BorderCapacitySum(),
        description='Sum of the line capacities across the border, in border direction.',
    )
    PRICE_SPREAD_SPEC = BorderFlagSpec(
        name='{border_layer}.Results.price_spread',
        base=DummyAreaFlags.PRICE_SPEC,
        aggregation=PriceSpread(),
        description='Area price at area_to minus the area price at area_from.',
    )
    flag_specs = (MODEL_SPEC, FLOW_SPEC, CAPACITY_SPEC, PRICE_SPREAD_SPEC)


class DummyPeriodDataset(DummyDataset):
    """A `DummyDataset` with a time level of (run, period) tuples instead of a DatetimeIndex.

    It inherits the interpreters of `DummyDataset`. See `DummyPeriodSource` for
    the run and period numbers.
    """

    SOURCE_TYPE: ClassVar[type[DummySource]] = DummyPeriodSource


def make_dummy_dataset(scenario: DummyScenario | tuple[DummyScenario, ...], name: str | None = None) -> Dataset:
    """Build the dataset of one scenario or of a scenario split into parts.

    Args:
        scenario: One scenario, or the parts of `DummyScenario.split`.
        name: Dataset name. Default: the scenario name, or for parts their
            name without the '.N' suffix ('base' for 'base.1' and 'base.2').

    Returns:
        A `DummyDataset` for one scenario. A `DatasetMergeCollection` of one
            `DummyDataset` per part for parts, with the attributes of the whole:
            'first_date' of the first part, 'last_date' of the last part and the
            sum of 'days'.

    Raises:
        ValueError: If no part is given, or if the parts have different names
            without the suffix and no `name` is given.
    """
    if isinstance(scenario, DummyScenario):
        return DummyDataset(scenario, name=name)
    if not scenario:
        raise ValueError('make_dummy_dataset needs at least one part.')
    parts = [DummyDataset(part) for part in scenario]
    if name is None:
        base_names = {re.sub(r'\.\d+$', '', part.name) for part in parts}
        if len(base_names) != 1:
            raise ValueError(f'Parts {[p.name for p in parts]} have no common name. Pass `name`.')
        name = base_names.pop()
    attributes = {
        'first_date': parts[0].attributes['first_date'],
        'last_date': parts[-1].attributes['last_date'],
        'days': sum(part.attributes['days'] for part in parts),
    }
    return DatasetMergeCollection(parts, name=name, attributes=attributes)
