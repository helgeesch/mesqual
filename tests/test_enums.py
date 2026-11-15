from mesqual.enums import TopologyTypeEnum, VisualizationTypeEnum


def test_every_topology_type_has_a_visualization_type():
    assert all(isinstance(t.visualization_type, VisualizationTypeEnum) for t in TopologyTypeEnum)


def test_border_is_shown_as_border():
    assert TopologyTypeEnum.Border.visualization_type is VisualizationTypeEnum.Border
