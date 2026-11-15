"""A small platform with hand-written tables, for hand-computed area and border values."""

from __future__ import annotations

from typing import ClassVar

import pandas as pd

from mesqual.datasets import Dataset, PlatformDataset, SourceInventory
from mesqual.layer_flags import (
    AreaFlagInterpreter,
    AreaModel,
    BorderCapacitySum,
    BorderFlagInterpreter,
    BorderFlowSum,
    BorderModel,
    PriceSpread,
    Sum,
    WeightedMean,
)
from mesqual.enums import TopologyTypeEnum
from mesqual.flag import (
    AreaFlagSpec,
    AreaLayer,
    BorderFlagSpec,
    BorderLayer,
    Dimension,
    ModelFlagSpec,
    TimeSeriesFlagSpec,
)
from mesqual.units import Units

TIME = pd.date_range('2024-01-01', periods=2, freq='h', name='time')
NODES = pd.DataFrame({'zone': ['A', 'A', 'B', '']}, index=pd.Index(['n1', 'n2', 'n3', 'n4'], name='Node'))
"""n4 has no zone."""
LINES = pd.DataFrame(
    {'node_from': ['n1', 'n3', 'n1', 'n3'], 'node_to': ['n3', 'n2', 'n2', 'n4']},
    index=pd.Index(['l1', 'l2', 'l3', 'l4'], name='Line'),
)
"""l1 runs A to B, l2 B to A, l3 inside A, l4 from B to n4 without zone."""


def frame(columns: dict) -> pd.DataFrame:
    return pd.DataFrame(columns, index=TIME)


PRICES = {
    'peak': frame({'n1': [10.0, 20.0], 'n2': [30.0, 40.0], 'n3': [50.0, 60.0], 'n4': [70.0, 80.0]}),
    'offpeak': frame({'n1': [11.0, 21.0], 'n2': [31.0, 41.0], 'n3': [51.0, 61.0], 'n4': [71.0, 81.0]}),
}
DEMAND = {
    'peak': frame({'n1': [1.0, 1.0], 'n2': [3.0, 1.0], 'n3': [1.0, 1.0], 'n4': [1.0, 1.0]}),
    'offpeak': frame({'n1': [1.0, 2.0], 'n2': [1.0, 2.0], 'n3': [1.0, 1.0], 'n4': [1.0, 1.0]}),
}
UP = frame({'l1': [100.0, 100.0], 'l2': [10.0, 10.0], 'l3': [7.0, 7.0], 'l4': [3.0, 3.0]})
DOWN = frame({'l1': [80.0, 80.0], 'l2': [5.0, 5.0], 'l3': [6.0, 6.0], 'l4': [2.0, 2.0]})
CAPACITIES = {'up': UP, 'down': DOWN, 'net': UP - DOWN, 'total': UP + DOWN, 'unknown': UP}
"""Line capacities, and line flows with the same values."""

PERIOD = Dimension('period', 'Load period.', {'peak': 'peak hours', 'offpeak': 'offpeak'})
NODE_MODEL_SPEC = ModelFlagSpec(
    name='Node.Model',
    index='Node',
    topology_type=TopologyTypeEnum.Node,
    description='Nodes.',
    memberships={'zone': 'Zone.Model'},
)
LINE_MODEL_SPEC = ModelFlagSpec(
    name='Line.Model',
    index='Line',
    topology_type=TopologyTypeEnum.Edge,
    description='Lines.',
    memberships={'node_from': 'Node.Model', 'node_to': 'Node.Model'},
    edge_ends=('node_from', 'node_to'),
)
PRICE_SPEC = TimeSeriesFlagSpec(
    name='Node.Results.price_{period}',
    description='Node price.',
    model_flag='Node.Model',
    unit=Units.EUR_per_MWh,
    variants=(PERIOD,),
)
DEMAND_SPEC = TimeSeriesFlagSpec(
    name='Node.Results.demand_{period}',
    description='Node demand.',
    model_flag='Node.Model',
    unit=Units.MW,
    variants=(PERIOD,),
    fill_value=0,
)
CAPACITY_SPEC = TimeSeriesFlagSpec(
    name='Line.Constraints.capacity_{direction}',
    description='Line capacity.',
    model_flag='Line.Model',
    unit=Units.MW,
    fill_value=0,
    variants=(Dimension('direction', 'Direction.', {d: d for d in CAPACITIES}),),
)
FLOW_SPEC = TimeSeriesFlagSpec(
    name='Line.Results.flow_{direction}',
    description='Line flow.',
    model_flag='Line.Model',
    unit=Units.MW,
    fill_value=0,
    variants=(Dimension('direction', 'Direction.', {d: d for d in CAPACITIES}),),
)
ZONE = AreaLayer(name='Zone', index='zone', base_model='Node.Model', layer_model='Zone.Model', description='Zones.')
ZONE_BORDER = BorderLayer(
    name='ZoneBorder',
    index='zone_border',
    area_layer='Zone',
    edge_model='Line.Model',
    description='Zone borders.',
)
REGION = AreaLayer(
    name='Region',
    index='region',
    base_model='Node.Model',
    layer_model='Region.Model',
    description='Regions.',
)

AREA_PRICE = AreaFlagSpec(
    name='{area_layer}.Results.price_{period}',
    base=PRICE_SPEC,
    aggregation=WeightedMean(weight='Node.Results.demand_{period}'),
    description='Area price.',
)
AREA_DEMAND = AreaFlagSpec(
    name='{area_layer}.Results.demand_{period}',
    base=DEMAND_SPEC,
    aggregation=Sum(),
    description='Area demand.',
)
AREA_MODEL = AreaFlagSpec(
    name='{area_layer}.Model', base=NODE_MODEL_SPEC, aggregation=AreaModel(), description='Areas.'
)
BORDER_MODEL_SPEC = BorderFlagSpec(
    name='{border_layer}.Model',
    base=LINE_MODEL_SPEC,
    aggregation=BorderModel(geometry=False),
    description='Borders.',
)
SPREAD = BorderFlagSpec(
    name='{border_layer}.Results.spread_{period}',
    base=AREA_PRICE,
    aggregation=PriceSpread(),
    description='Spread.',
)
BORDER_FLOW = BorderFlagSpec(
    name='{border_layer}.Results.flow_{direction}',
    base=FLOW_SPEC,
    aggregation=BorderFlowSum(),
    description='Border flow.',
)
BORDER_CAPACITY = BorderFlagSpec(
    name='{border_layer}.Constraints.capacity_{direction}',
    base=CAPACITY_SPEC,
    aggregation=BorderCapacitySum(),
    description='Border capacity.',
)
AREA_EXCHANGE = AreaFlagSpec(
    name='{area_layer}.Results.exchange',
    base=BORDER_FLOW,
    aggregation=Sum(),
    description='Exchange.',
    consumes=('direction',),
)
"""An area spec on a border base."""


class TableInterpreter(Dataset):
    """Base of the interpreters of `TablePlatform`: answers its flags from `tables`."""

    tables: ClassVar[dict] = {}

    def __init__(self, parent_dataset: Dataset | None = None):
        super().__init__(parent_dataset=parent_dataset)

    def _fetch(self, flag, effective_config, **params):
        return self.tables[flag].copy()


class TableModels(TableInterpreter):
    """Node and line model."""

    flag_specs = (NODE_MODEL_SPEC, LINE_MODEL_SPEC)
    tables = {'Node.Model': NODES, 'Line.Model': LINES}


class TableResults(TableInterpreter):
    """Node prices and demands, line capacities and flows."""

    flag_specs = (PRICE_SPEC, DEMAND_SPEC, CAPACITY_SPEC, FLOW_SPEC)
    tables = {
        **{f'Node.Results.price_{m}': df for m, df in PRICES.items()},
        **{f'Node.Results.demand_{m}': df for m, df in DEMAND.items()},
        **{f'Line.Constraints.capacity_{d}': df for d, df in CAPACITIES.items()},
        **{f'Line.Results.flow_{d}': df for d, df in CAPACITIES.items()},
    }


class TablePlatform(PlatformDataset):
    """Platform with the base interpreters and no layers."""

    def build_source_inventory(self) -> SourceInventory:
        return SourceInventory(fingerprint='tables')

    @classmethod
    def get_child_dataset_type(cls):
        return TableInterpreter


TablePlatform.register_interpreter(TableModels)
TablePlatform.register_interpreter(TableResults)


def platform(
    area_specs: tuple = (),
    border_specs: tuple = (),
    area_layers: tuple = (ZONE,),
    border_layers: tuple = (ZONE_BORDER,),
    base: type[PlatformDataset] = TablePlatform,
) -> type[PlatformDataset]:
    """A new subclass of `base` with the given layers and one area and one border interpreter."""
    cls = type('Layered', (base,), {})
    for layer in area_layers:
        cls.register_area_layer(layer)
    for layer in border_layers:
        cls.register_border_layer(layer)
    if area_specs:
        cls.register_interpreter(type('Areas', (AreaFlagInterpreter,), {'flag_specs': tuple(area_specs)}))
    if border_specs:
        cls.register_interpreter(type('Borders', (BorderFlagInterpreter,), {'flag_specs': tuple(border_specs)}))
    return cls
