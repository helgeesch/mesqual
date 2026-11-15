"""Example specs for the flag tests."""

from mesqual.enums import TopologyTypeEnum
from mesqual.flag import Dimension, FrameSchema, Level, ModelFlagSpec, TimeSeriesFlagSpec
from mesqual.units import Units

SIDE = Dimension(
    'side', 'Where the flow is measured.', {'sent': 'where the flow enters', 'received': 'where the flow leaves'}
)
DIRECTION = Dimension('direction', 'Flow direction.', {'up': 'in line direction', 'net': 'up minus down'})
SOLUTION = Dimension(
    'solution',
    'Solution of the optimisation to report.',
    {'final': 'final solution', 'first': 'first solution'},
    default='final',
)

NODE_MODEL_SPEC = ModelFlagSpec(
    name='Node.Model',
    description='Nodes.',
    index='Node',
    topology_type=TopologyTypeEnum.Node,
)
LINE_MODEL_SPEC = ModelFlagSpec(
    name='Line.Model',
    description='Lines between two nodes.',
    index='Line',
    topology_type=TopologyTypeEnum.Edge,
    memberships={'node_from': 'Node.Model', 'node_to': 'Node.Model'},
    edge_ends=('node_from', 'node_to'),
)
NODE_PRICE_SPEC = TimeSeriesFlagSpec(
    name='Node.Results.price',
    description='Clearing price at the node.',
    unit=Units.EUR_per_MWh,
    model_flag='Node.Model',
    params=(SOLUTION,),
)
LINE_FLOW_SPEC = TimeSeriesFlagSpec(
    name='Line.Results.flow_{side}_{direction}',
    description='Power flow on the line.',
    unit=Units.MW,
    model_flag='Line.Model',
    variants=(SIDE, DIRECTION),
)
SYSTEM_LOAD_SPEC = TimeSeriesFlagSpec(
    name='System.Results.load',
    description='Total load.',
    unit=Units.MW,
    model_flag=None,
    schema=FrameSchema(index=(Level('time', 'time'),)),
)
