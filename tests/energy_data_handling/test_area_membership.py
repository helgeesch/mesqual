"""Area membership: one rule for the area calculators, the model generators and the edge areas."""

import pandas as pd
import pytest

from mesqual.energy_data_handling.area_accounting import (
    AreaBorderModelGenerator,
    AreaMembership,
    AreaModelGenerator,
    AreaPriceCalculator,
    AreaSumCalculator,
)

NODES = pd.DataFrame(
    {'area': ['A', 'A', 'B', '', None, float('nan')]},
    index=['n1', 'n2', 'n3', 'n4', 'n5', 'n6'],
)
"""n4, n5 and n6 have no area."""
LINES = pd.DataFrame(
    {'node_from': ['n1', 'n3', 'n3'], 'node_to': ['n3', 'n4', 'n5']},
    index=['l1', 'l2', 'l3'],
)


def test_has_area_is_false_for_nan_none_and_an_empty_string():
    areas = pd.Series(['A', '', None, float('nan'), 0])
    assert AreaMembership.has_area(areas).tolist() == [True, False, False, False, True]


def test_the_area_calculators_leave_out_nodes_with_nan_none_or_an_empty_string_as_area():
    values = pd.DataFrame([[1.0, 3.0, 5.0, 100.0, 100.0, 100.0]], columns=NODES.index)
    sum_calculator = AreaSumCalculator(NODES, 'area')
    price_calculator = AreaPriceCalculator(NODES, 'area')
    assert sum_calculator.areas == price_calculator.areas == ['A', 'B']
    assert sum_calculator.get_area_nodes('A') == ['n1', 'n2']
    assert sum_calculator.get_area_nodes('') == []
    expected_sum = pd.DataFrame({'A': [4.0], 'B': [5.0]}).rename_axis(columns='area')
    expected_mean = pd.DataFrame({'A': [2.0], 'B': [5.0]}).rename_axis(columns='area')
    pd.testing.assert_frame_equal(sum_calculator.calculate(values), expected_sum)
    pd.testing.assert_frame_equal(price_calculator.calculate(values), expected_mean)


def test_a_node_without_area_adds_no_area_to_the_area_model():
    base_model = AreaModelGenerator(NODES, 'area').generate_base_area_model_from_area_names_in_node_model_df()
    assert base_model.index.tolist() == ['A', 'B']


def test_edge_areas_are_nan_for_an_end_without_area_and_for_an_unknown_node():
    lines = pd.DataFrame({'node_from': ['n3', 'n1'], 'node_to': ['n4', 'unknown']})
    assert AreaMembership.edge_areas(lines, NODES['area'], 'node_from', 'node_to')['area_to'].isna().all()


def test_edge_areas_with_duplicate_node_ids_raise_value_error_that_names_them():
    nodes = pd.Series(['A', 'B', 'C'], index=['n1', 'n2', 'n2'])
    with pytest.raises(ValueError, match=r"\['n2'\]"):
        AreaMembership.edge_areas(LINES, nodes, 'node_from', 'node_to')


def test_integer_area_ids_including_zero_give_borders_also_with_a_line_to_an_unknown_node():
    nodes = pd.DataFrame({'area': [0, 1, 1]}, index=['n1', 'n2', 'n3'])
    lines = pd.DataFrame(
        {'node_from': ['n1', 'n2', 'n3'], 'node_to': ['n2', 'n3', 'unknown']},
        index=['l1', 'l2', 'l3'],
    )
    model = AreaBorderModelGenerator(nodes, lines, 'area', 'node_from', 'node_to').generate_area_border_model()
    assert model.index.tolist() == ['0 - 1', '1 - 0']


@pytest.mark.parametrize(
    'build',
    [
        lambda nodes: AreaSumCalculator(nodes, 'zone'),
        lambda nodes: AreaModelGenerator(nodes, 'zone'),
        lambda nodes: AreaBorderModelGenerator(nodes, LINES, 'zone', 'node_from', 'node_to'),
    ],
    ids=['area calculator', 'area model generator', 'border model generator'],
)
def test_a_missing_area_column_raises_value_error_with_the_available_columns(build):
    with pytest.raises(ValueError, match=r"Area column 'zone' not found in node_model_df. Available columns: \['x'\]"):
        build(pd.DataFrame({'x': [1]}, index=['n']))
