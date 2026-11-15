"""Border naming: the canonical border has the area names in alphabetical order."""

import pandas as pd

from mesqual.energy_data_handling.area_accounting import AreaBorderModelGenerator, AreaBorderNamingConventions


def test_a_border_and_its_opposite_have_the_same_canonical_border():
    conventions = AreaBorderNamingConventions('zone')
    assert conventions.get_canonical_border('FR - DE') == 'DE - FR'
    assert conventions.get_canonical_border('DE - FR') == 'DE - FR'


def test_border_model_has_the_canonical_border_and_is_canonical():
    nodes = pd.DataFrame({'zone': ['FR', 'DE']}, index=['n1', 'n2'])
    lines = pd.DataFrame({'node_from': ['n1'], 'node_to': ['n2']}, index=['l1'])
    model = AreaBorderModelGenerator(nodes, lines, 'zone', 'node_from', 'node_to').generate_area_border_model()
    assert model['canonical_border'].to_dict() == {
        'DE - FR': 'DE - FR',
        'FR - DE': 'DE - FR',
    }
    assert model['is_canonical'].to_dict() == {'DE - FR': True, 'FR - DE': False}


def test_empty_border_model_has_the_border_index_and_the_canonical_columns():
    nodes = pd.DataFrame({'zone': ['DE', 'DE']}, index=['n1', 'n2'])
    lines = pd.DataFrame({'node_from': ['n1'], 'node_to': ['n2']}, index=['l1'])
    model = AreaBorderModelGenerator(nodes, lines, 'zone', 'node_from', 'node_to').generate_area_border_model()
    assert model.index.name == 'zone_border'
    assert list(model.columns) == ['area_from', 'area_to', 'opposite_border', 'canonical_border', 'is_canonical']
    assert model['is_canonical'].dtype == bool


def test_border_model_has_the_given_area_column_names_and_the_defaults_for_none():
    nodes = pd.DataFrame({'zone': ['FR', 'DE']}, index=['n1', 'n2'])
    lines = pd.DataFrame({'node_from': ['n1'], 'node_to': ['n2']}, index=['l1'])
    custom = AreaBorderModelGenerator(
        nodes,
        lines,
        'zone',
        'node_from',
        'node_to',
        source_area_identifier='zone_from',
        target_area_identifier='zone_to',
    )
    assert list(custom.line_areas.columns) == ['zone_from', 'zone_to']
    assert list(custom.generate_area_border_model().columns[:2]) == ['zone_from', 'zone_to']
    default = AreaBorderModelGenerator(
        nodes,
        lines,
        'zone',
        'node_from',
        'node_to',
        source_area_identifier=None,
        target_area_identifier=None,
    )
    assert list(default.line_areas.columns) == ['area_from', 'area_to']
    assert list(default.generate_area_border_model().columns[:2]) == ['area_from', 'area_to']
