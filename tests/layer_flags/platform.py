"""Platform with two area layers and their border layers on the synthetic scenario data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import pandas as pd

from mesqual.datasets.platform_dataset import PlatformDataset
from mesqual.layer_flags import (
    AreaFlagInterpreter,
    AreaModel,
    AreaAggregation,
    BorderCapacitySum,
    BorderFlagInterpreter,
    BorderFlowSum,
    BorderModel,
    PriceSpread,
    Sum,
    WeightedMean,
)
from mesqual.energy_data_handling.area_accounting import AreaMembership
from mesqual.energy_data_handling.variable_utils import trade_balance_frame
from mesqual.enums import TopologyTypeEnum
from mesqual.flag import (
    AreaFlagSpec,
    AreaLayer,
    BorderFlagSpec,
    BorderLayer,
    Dimension,
    FlagIndex,
    FrameSchema,
    Level,
    ModelFlagSpec,
    TimeSeriesFlagSpec,
)
from mesqual.units import Units
from mesqual.testing import DummyConfig, DummyScenario, DummySource
from mesqual.testing.source import DummyInventory
from mesqual.testing.dataset import (
    DummyDerivedInterpreter,
    DummyInterpreter,
    DummyModelInterpreter,
    DummyTimeSeriesInterpreter,
)

NODE_MODEL_SPEC = ModelFlagSpec(
    name='Node.Model',
    index='Node',
    topology_type=TopologyTypeEnum.Node,
    description='Nodes of the network.',
    memberships={'country': 'Country.Model', 'macroregion': 'Macroregion.Model'},
)
OTHER_MODEL_SPECS = tuple(s for s in DummyModelInterpreter.flag_specs if s.name not in ('Node.Model', 'Country.Model'))
LINE_MODEL_SPEC = next(s for s in OTHER_MODEL_SPECS if s.name == 'Line.Model')
FLOW_SPEC = DummyTimeSeriesInterpreter.FLOW_SPEC
CAPACITY_SPEC = DummyTimeSeriesInterpreter.CAPACITY_SPEC
PRICE_SPEC = DummyTimeSeriesInterpreter.PRICE_SPEC

MACROREGION_OF_COUNTRY = {'C1': 'North', 'C2': 'South'}
MOVED_NODE = 'N4'
"""Node in the macroregion North, though its country C2 lies in South."""

COUNTRY = AreaLayer(
    name='Country',
    index='country',
    base_model='Node.Model',
    layer_model='Country.Model',
    description='Countries of the nodes.',
)
MACROREGION = AreaLayer(
    name='Macroregion',
    index='macroregion',
    base_model='Node.Model',
    layer_model='Macroregion.Model',
    description='Custom regions of the nodes, for example the Nordic countries.',
)
COUNTRY_BORDER = BorderLayer(
    name='CountryBorder',
    index='country_border',
    area_layer='Country',
    edge_model='Line.Model',
    description='Borders between two countries, crossed by lines.',
)
MACROREGION_BORDER = BorderLayer(
    name='MacroregionBorder',
    index='macroregion_border',
    area_layer='Macroregion',
    edge_model='Line.Model',
    description='Borders between two macroregions, crossed by lines.',
)

AREA_MODEL_SPEC = AreaFlagSpec(
    name='{area_layer}.Model',
    base=NODE_MODEL_SPEC,
    aggregation=AreaModel(),
    description='One row per area.',
)
AREA_PRICE_SPEC = AreaFlagSpec(
    name='{area_layer}.Results.price',
    base=PRICE_SPEC,
    aggregation=WeightedMean(weight='Node.Results.demand_total'),
    description='Demand-weighted mean of the node prices of the area.',
)
AREA_DEMAND_SPEC = AreaFlagSpec(
    name='{area_layer}.Results.demand_{product}',
    base=DummyDerivedInterpreter.NODE_DEMAND_SPEC,
    aggregation=Sum(),
    description='Sum of the node demand of the area.',
)
BORDER_MODEL_SPEC = BorderFlagSpec(
    name='{border_layer}.Model',
    base=LINE_MODEL_SPEC,
    aggregation=BorderModel(geometry=False),
    description='One row per border: area_from, area_to, the lines in and against border direction.',
)
BORDER_FLOW_SPEC = BorderFlagSpec(
    name='{border_layer}.Results.flow_{side}_{direction}',
    base=FLOW_SPEC,
    aggregation=BorderFlowSum(),
    description='Sum of the line flows across the border, in border direction.',
)
BORDER_CAPACITY_SPEC = BorderFlagSpec(
    name='{border_layer}.Constraints.capacity_{direction}',
    base=CAPACITY_SPEC,
    aggregation=BorderCapacitySum(),
    description='Sum of the line capacities across the border, in border direction.',
)
BORDER_PRICE_SPREAD_SPEC = BorderFlagSpec(
    name='{border_layer}.Results.price_spread',
    base=AREA_PRICE_SPEC,
    aggregation=PriceSpread(),
    description='Area price at area_to minus the price at area_from.',
)

LOSSES = Dimension(
    'losses',
    'How line losses enter the trade balance.',
    {
        'at_own_end': 'export before losses, import after losses',
        'excluded': 'export and import before losses',
    },
    default='at_own_end',
)


@dataclass(frozen=True, kw_only=True)
class TradeBalancePerPartner(AreaAggregation):
    """Net export of each area to each partner, from the border flows."""

    SIDES: ClassVar[tuple[str, ...]] = ('sent', 'received')
    DIRECTIONS: ClassVar[tuple[str, ...]] = ('up', 'down')

    def input_flags(self, dataset, layer, variant):
        return tuple(variant.base_flag(side=s, direction=d) for s in self.SIDES for d in self.DIRECTIONS)

    def aggregate(self, values, mapping, dataset, layer, variant, effective_config, losses, **params):
        flows = {
            (s, d): self.fetch(dataset, variant.base_flag(side=s, direction=d), effective_config, params)
            for s in self.SIDES
            for d in self.DIRECTIONS
        }
        import_side = {'at_own_end': 'received', 'excluded': 'sent'}[losses]
        net_exp = flows['sent', 'up'] - flows[import_side, 'down']
        net_exports = {
            (row[AreaMembership.AREA_FROM], row[AreaMembership.AREA_TO]): net_exp[border]
            for border, row in mapping.iterrows()
        }
        return trade_balance_frame(net_exports, layer.index, f'partner_{layer.index}', net_exp.index)


@dataclass(frozen=True, kw_only=True)
class TradeBalanceTotal(AreaAggregation):
    """Trade balance of each area, summed over its partners per variable."""

    def aggregate(self, values, mapping, dataset, layer, variant, effective_config, losses, **params):
        flag = f'{variant.area_layer}.Results.trade_balance_per_partner'
        per_partner = dataset.fetch(flag, effective_config, losses=losses, **params)
        return per_partner.T.groupby(level=[0, 2]).sum(min_count=1).T


TRADE_BALANCE_PER_PARTNER_SPEC = AreaFlagSpec(
    name='{area_layer}.Results.trade_balance_per_partner',
    base=BORDER_FLOW_SPEC,
    aggregation=TradeBalancePerPartner(),
    description='Trade balance of each area with each partner. net_exp is positive from the area to the partner.',
    consumes=('side', 'direction'),
    params=(LOSSES,),
    schema=lambda v, layer: FrameSchema(
        index=(Level('time', 'time'),),
        columns=(
            Level(layer.index, 'object'),
            Level(f'partner_{layer.index}', 'counterpart'),
            Level('variable', 'dimension'),
        ),
    ),
)
TRADE_BALANCE_SPEC = AreaFlagSpec(
    name='{area_layer}.Results.trade_balance',
    base=BORDER_FLOW_SPEC,
    aggregation=TradeBalanceTotal(),
    description='Trade balance of each area, summed over its partners per variable.',
    consumes=('side', 'direction'),
    params=(LOSSES,),
    schema=lambda v, layer: FrameSchema(
        index=(Level('time', 'time'),),
        columns=(Level(layer.index, 'object'), Level('variable', 'dimension')),
    ),
)


SCALED_FLOW_SPEC = TimeSeriesFlagSpec(
    name='Line.Results.scaled_flow_{direction}',
    description='Line flow where it enters the line, times a factor.',
    model_flag='Line.Model',
    unit=Units.MW,
    fill_value=0,
    variants=(FLOW_SPEC.variants[1],),
    params=(Dimension('scale', 'Factor on the flow.', {1: 'the flow', 2: 'twice the flow'}, default=1),),
)
BORDER_SCALED_FLOW_SPEC = BorderFlagSpec(
    name='{border_layer}.Results.scaled_flow_{direction}',
    base=SCALED_FLOW_SPEC,
    aggregation=BorderFlowSum(),
    description='Sum of the scaled line flows across the border, in border direction.',
)


class ScaledFlowInterpreter(DummyInterpreter):
    """A line flow with a base parameter, to test that parameters reach the base flag of a border flag."""

    flag_specs = (SCALED_FLOW_SPEC,)
    INPUT_FLAGS: ClassVar[dict[str, tuple[str, ...]]] = {SCALED_FLOW_SPEC.name: ('Line.Results.flow_sent_{direction}',)}

    def _fetch(self, flag: str, effective_config: DummyConfig, scale: int, **params) -> pd.DataFrame:
        [flow] = self._fetch_inputs(flag, effective_config, params)
        return flow * scale


class LayerModelInterpreter(DummyInterpreter):
    """Model tables, with `Node.Model` in two country layers."""

    flag_specs = (NODE_MODEL_SPEC,) + OTHER_MODEL_SPECS
    SPEC_TABLE: ClassVar[dict[str, str]] = {
        'Node.Model': 'nodes',
        'Line.Model': 'lines',
        'Generator.Model': 'generators',
        'Demand.Model': 'demands',
    }

    def _fetch(self, flag: str, effective_config: DummyConfig, **params) -> pd.DataFrame:
        table = self.source.read(self.SPEC_TABLE[flag])
        if flag != 'Node.Model':
            return table
        macroregion = table['country'].map(MACROREGION_OF_COUNTRY).where(table.index != MOVED_NODE, 'North')
        return table.assign(macroregion=macroregion)


class LayerPlatform(PlatformDataset[DummyInterpreter, DummyConfig, str, FlagIndex]):
    """Platform on the synthetic scenario data with the layers Country and Macroregion and their border layers."""

    def __init__(self, scenario: DummyScenario | None = None, name: str | None = None, **kwargs):
        scenario = scenario or DummyScenario()
        self.source = DummySource(scenario)
        super().__init__(name=name or scenario.name, source=self.source, **kwargs)

    def build_source_inventory(self) -> DummyInventory:
        return self.source.inventory

    @classmethod
    def get_config_type(cls) -> type[DummyConfig]:
        return DummyConfig

    @classmethod
    def get_child_dataset_type(cls) -> type[DummyInterpreter]:
        return DummyInterpreter


for interpreter in (LayerModelInterpreter, DummyTimeSeriesInterpreter, DummyDerivedInterpreter, ScaledFlowInterpreter):
    LayerPlatform.register_interpreter(interpreter)
for area_layer in (COUNTRY, MACROREGION):
    LayerPlatform.register_area_layer(area_layer)
for border_layer in (COUNTRY_BORDER, MACROREGION_BORDER):
    LayerPlatform.register_border_layer(border_layer)


@LayerPlatform.register_interpreter
class LayerAreaFlags(AreaFlagInterpreter):
    """Area model, area price, area demand and trade balances."""

    flag_specs = (
        AREA_MODEL_SPEC,
        AREA_PRICE_SPEC,
        AREA_DEMAND_SPEC,
        TRADE_BALANCE_PER_PARTNER_SPEC,
        TRADE_BALANCE_SPEC,
    )


@LayerPlatform.register_interpreter
class LayerBorderFlags(BorderFlagInterpreter):
    """Border model, flows, capacities and price spread."""

    flag_specs = (
        BORDER_MODEL_SPEC,
        BORDER_FLOW_SPEC,
        BORDER_CAPACITY_SPEC,
        BORDER_PRICE_SPREAD_SPEC,
        BORDER_SCALED_FLOW_SPEC,
    )
