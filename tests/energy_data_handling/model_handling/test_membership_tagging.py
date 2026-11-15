"""Membership enrichers: tagging of the new columns and references to objects that the model does not have."""

import logging

import pandas as pd
import pytest

from mesqual.energy_data_handling.model_handling.membership_property_enrichers import (
    DirectionalMembershipPropertyEnricher,
    MembershipPropertyEnricher,
    MembershipTagging,
)
from mesqual.testing import DummyDataset, DummyScenario


@pytest.fixture(scope='module')
def dataset() -> DummyDataset:
    return DummyDataset(DummyScenario())


def voltage_of(dataset: DummyDataset, nodes: pd.Series) -> pd.Series:
    return nodes.map(dataset.fetch('Node.Model')['voltage'])


@pytest.mark.parametrize(
    'tagging, column',
    [
        (MembershipTagging.NONE, 'voltage'),
        (MembershipTagging.PREFIX, 'node_voltage'),
        (MembershipTagging.SUFFIX, 'voltage_node'),
    ],
)
def test_tagging_names_the_property_column(dataset, tagging, column):
    generators = dataset.fetch('Generator.Model')
    result = MembershipPropertyEnricher().append_properties(generators, dataset, 'Generator.Model', tagging)
    pd.testing.assert_series_equal(result[column], voltage_of(dataset, generators['node']), check_names=False)


def test_reference_to_an_object_missing_in_the_model_gives_nan_and_logs_a_warning(dataset, caplog):
    generators = dataset.fetch('Generator.Model')
    generators.loc['G1', 'node'] = 'N9'
    with caplog.at_level(logging.WARNING):
        result = MembershipPropertyEnricher().append_properties(generators, dataset, 'Generator.Model')
    assert pd.isna(result.loc['G1', 'voltage'])
    assert result['voltage'].drop('G1').notna().all()
    assert "1 objects missing in source dataframe for node: ['N9']" in caplog.text


@pytest.mark.parametrize(
    'tagging, columns',
    [
        (MembershipTagging.NONE, ['voltage_from', 'voltage_to']),
        (MembershipTagging.PREFIX, ['node_voltage_from', 'node_voltage_to']),
        (MembershipTagging.SUFFIX, ['voltage_node_from', 'voltage_node_to']),
    ],
)
def test_directional_enricher_tags_both_ends(dataset, tagging, columns, caplog):
    lines = dataset.fetch('Line.Model')
    lines.loc['L1', 'node_to'] = 'N9'
    with caplog.at_level(logging.WARNING):
        result = DirectionalMembershipPropertyEnricher().append_properties(lines, dataset, 'Line.Model', tagging)
    for column, end in zip(columns, ['node_from', 'node_to']):
        pd.testing.assert_series_equal(result[column], voltage_of(dataset, lines[end]), check_names=False)
    assert pd.isna(result.loc['L1', columns[1]])
    assert "for node_to: ['N9']" in caplog.text
