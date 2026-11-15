import pandas as pd
import pytest

from mesqual.testing import DummyNetwork, DummyScenario


def test_default_network_has_two_countries_four_nodes_five_lines_five_generators_eight_demands():
    network = DummyNetwork(DummyScenario())
    tables = (network.countries, network.nodes, network.lines, network.generators, network.demands)
    assert [len(t) for t in tables] == [2, 4, 5, 5, 8]
    assert network.countries.index.tolist() == ['C1', 'C2']
    assert network.nodes['country'].tolist() == ['C1', 'C2', 'C1', 'C2']
    assert set(network.lines.node_from) | set(network.lines.node_to) <= set(network.nodes.index)
    assert set(network.generators.node) <= set(network.nodes.index)
    assert network.generators['fuel'].tolist() == list(DummyNetwork.FUELS)
    assert set(network.demands.node) == set(network.nodes.index)
    assert [t.index.name for t in tables] == ['Country', 'Node', 'Line', 'Generator', 'Demand']


def test_each_node_lies_in_the_polygon_of_its_country():
    network = DummyNetwork(DummyScenario(n_countries=3, n_nodes=7, n_lines=7))
    for node in network.nodes.itertuples():
        assert network.countries.loc[node.country, 'geometry'].contains(node.location)


def test_node_without_country_has_no_country_no_lines_and_no_demands():
    network = DummyNetwork(DummyScenario(node_without_country=True))
    assert pd.isna(network.nodes.loc['N0', 'country'])
    assert 'N0' not in set(network.lines.node_from) | set(network.lines.node_to)
    assert 'N0' not in set(network.demands.node)


def test_removed_node_removes_its_lines_generators_and_demands():
    network = DummyNetwork(DummyScenario(removed_objects=('N1',)))
    assert 'N1' not in network.nodes.index
    assert 'N1' not in set(network.lines.node_from) | set(network.lines.node_to)
    assert 'N1' not in set(network.generators.node)
    assert network.demands.index.tolist() == ['D3', 'D4', 'D5', 'D6', 'D7', 'D8']


def test_text_changes_change_the_property():
    network = DummyNetwork(DummyScenario(text_changes=(('generators', 'G1', 'fuel', 'gas'),)))
    assert network.generators.loc['G1', 'fuel'] == 'gas'


def test_text_change_of_an_unknown_object_raises():
    with pytest.raises(KeyError):
        DummyNetwork(DummyScenario(text_changes=(('generators', 'G99', 'fuel', 'gas'),)))


def test_more_lines_than_node_pairs_raises():
    with pytest.raises(ValueError):
        DummyNetwork(DummyScenario(n_nodes=3, n_countries=1, n_lines=4))


def test_more_nodes_only_append_rows():
    small = DummyNetwork(DummyScenario(n_nodes=4, n_generators=6))
    large = DummyNetwork(DummyScenario(n_nodes=5, n_lines=7, n_generators=6))
    pd.testing.assert_frame_equal(large.nodes.iloc[:4], small.nodes)
    pd.testing.assert_frame_equal(large.lines.iloc[:5], small.lines)
    pd.testing.assert_frame_equal(large.generators, small.generators)
    pd.testing.assert_frame_equal(large.demands, small.demands)
    pd.testing.assert_frame_equal(large.countries, small.countries)


def test_more_demands_only_append_rows():
    small = DummyNetwork(DummyScenario())
    large = DummyNetwork(DummyScenario(n_nodes=5, n_demands=10))
    pd.testing.assert_frame_equal(large.demands.iloc[:8], small.demands)
    assert large.demands.loc[['D9', 'D10'], 'node'].tolist() == ['N5', 'N5']


def test_more_generators_than_node_capacity_raises():
    with pytest.raises(ValueError):
        DummyNetwork(DummyScenario(n_nodes=4, n_generators=9))


def test_more_demands_than_node_capacity_raises():
    with pytest.raises(ValueError):
        DummyNetwork(DummyScenario(n_nodes=4, n_demands=9))


def test_removed_country_leaves_its_nodes_without_country():
    network = DummyNetwork(DummyScenario(removed_objects=('C2',)))
    assert network.countries.index.tolist() == ['C1']
    assert network.nodes['country'].isna().tolist() == [False, True, False, True]


def test_scenario_without_country_raises():
    with pytest.raises(ValueError):
        DummyScenario(n_countries=0)
