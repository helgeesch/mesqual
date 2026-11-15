import pandas as pd

from mesqual.energy_data_handling.model_handling.membership_property_enrichers import (
    DirectionalMembershipPropertyEnricher,
    MembershipPropertyEnricher,
    MembershipTagging,
)
from mesqual.enums import TopologyTypeEnum
from mesqual.flag import ModelFlagSpec
from tests.datasets.fixtures import StaticDataset

NODE_SPEC = ModelFlagSpec(name='Node.Model', description='Nodes.', index='Node', topology_type=TopologyTypeEnum.Node)
LINE_SPEC = ModelFlagSpec(
    name='Line.Model',
    description='Lines.',
    index='Line',
    topology_type=TopologyTypeEnum.Edge,
    memberships={'node_from': 'Node.Model', 'node_to': 'Node.Model'},
    edge_ends=('node_from', 'node_to'),
)
GENERATOR_SPEC = ModelFlagSpec(
    name='Generator.Model',
    description='Generators.',
    index='Generator',
    topology_type=TopologyTypeEnum.NodeConnectedElement,
    memberships={'node': 'Node.Model'},
)

NODES = pd.DataFrame({'zone': ['DE', 'FR']}, index=pd.Index(['n1', 'n2'], name='Node'))
LINES = pd.DataFrame({'node_from': ['n1'], 'node_to': ['n2']}, index=pd.Index(['l1'], name='Line'))
GENERATORS = pd.DataFrame(
    {'node': ['n1', 'n2'], 'fuel_type': ['gas', 'coal']}, index=pd.Index(['g1', 'g2'], name='Generator')
)


def dataset() -> StaticDataset:
    return StaticDataset(
        {'Node.Model': NODES, 'Line.Model': LINES, 'Generator.Model': GENERATORS},
        specs=(NODE_SPEC, LINE_SPEC, GENERATOR_SPEC),
    )


def test_enricher_uses_the_memberships_of_the_model_flag():
    enricher = MembershipPropertyEnricher()
    ds = dataset()
    assert enricher.identify_membership_columns('Generator.Model', GENERATORS.columns, ds) == ['node']
    result = enricher.append_properties(GENERATORS, ds, 'Generator.Model', MembershipTagging.PREFIX)
    assert result['node_zone'].tolist() == ['DE', 'FR']
    assert 'fuel_type_zone' not in result


def test_directional_enricher_uses_both_ends():
    enricher = DirectionalMembershipPropertyEnricher()
    ds = dataset()
    assert enricher.identify_from_to_columns('Line.Model', LINES.columns, ds) == ['node']
    result = enricher.append_properties(LINES, ds, 'Line.Model')
    assert result[['zone_from', 'zone_to']].iloc[0].tolist() == ['DE', 'FR']
