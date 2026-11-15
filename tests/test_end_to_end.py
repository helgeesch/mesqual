import pandas as pd
import pytest

from mesqual import StudyManager
from mesqual.testing import DummyDataset, DummyScenario, make_dummy_dataset

BASE = DummyScenario(
    name='base',
    start='2024-03-30',
    days=3,
    tz='Europe/Berlin',
    resolution_change=('2024-03-31', '15min'),
    profile='constant',
)


@pytest.fixture(scope='module')
def study() -> StudyManager:
    high = BASE.replace(
        name='high_demand',
        demand_factor=1.2,
        n_generators=6,
        text_changes=(('generators', 'G2', 'fuel', 'gas'),),
    )
    short = BASE.replace(name='short', days=2)
    no_capacity = BASE.replace(name='no_capacity', missing_tables=('line_capacity',))
    return StudyManager.factory_from_scenarios(
        scenarios=[
            # Split base into two consecutive periods and merge them into one dataset.
            make_dummy_dataset(BASE.split('2024-03-31')),
            DummyDataset(high),
            DummyDataset(short),
            DummyDataset(no_capacity),
        ],
        comparisons=[('high_demand', 'base'), ('short', 'base')],
    )


def test_study_has_all_scenarios_and_comparisons(study):
    assert study.scenario_names == ['base', 'high_demand', 'short', 'no_capacity']
    assert study.comparison_names == ['high_demand vs base', 'short vs base']


def test_scenario_time_series_concat_with_differing_horizons(study):
    prices = study.scen.fetch('Node.Results.price')
    assert prices.columns.names == ['dataset', 'Node']
    assert prices.index.is_monotonic_increasing
    assert len(prices) == 24 + 92 + 96
    assert prices['short'].loc['2024-04-01'].isna().all().all()
    assert prices['base'].notna().all().all()


def test_split_scenario_equals_the_unsplit_scenario(study):
    split_flows = study.scen.get_dataset('base').fetch('Line.Results.flow_received_net')
    pd.testing.assert_frame_equal(split_flows, DummyDataset(BASE).fetch('Line.Results.flow_received_net'))


def test_flag_missing_in_one_scenario_leaves_it_out(study):
    capacities = study.scen.fetch('Line.Constraints.capacity_up')
    assert 'no_capacity' not in capacities.columns.get_level_values('dataset')
    assert 'base' in capacities.columns.get_level_values('dataset')


def test_comparison_of_time_series_gives_hand_computed_deltas(study):
    deltas = study.comp.fetch('Node.Results.demand_total')['high_demand vs base']
    base_demand = study.scen.get_dataset('base').fetch('Demand.Model').groupby('node')['base_demand'].sum()
    expected = pd.DataFrame([0.2 * base_demand] * len(deltas), index=deltas.index)
    pd.testing.assert_frame_equal(deltas, expected, check_names=False)


def test_comparison_of_model_shows_changed_and_new_objects(study):
    generators = study.comp.fetch('Generator.Model')['high_demand vs base']
    assert generators.loc['G2', 'fuel'] == 'gas (was coal)'
    assert generators.loc['G6', 'fuel'] == 'nuclear (new)'


def test_scenarios_and_comparisons_in_one_collection(study):
    demand = study.scen_comp.fetch('Node.Results.demand_total')
    assert demand.columns.names == ['type', 'dataset', 'Node']
    assert set(demand.columns.get_level_values('type')) == {'scenario', 'comparison'}
    pd.testing.assert_frame_equal(demand['scenario'], study.scen.fetch('Node.Results.demand_total'))
    pd.testing.assert_frame_equal(demand['comparison'], study.comp.fetch('Node.Results.demand_total'))


def test_comparison_outside_the_shorter_horizon_follows_the_fill_value(study):
    base_demand = study.scen.get_dataset('base').fetch('Node.Results.demand_total')
    filled = study.comp.fetch('Node.Results.demand_total')['short vs base']
    pd.testing.assert_frame_equal(filled.loc['2024-04-01'], -base_demand.loc['2024-04-01'], check_names=False)
    assert (filled.loc[:'2024-03-31'] == 0).all().all()
    unfilled = study.comp.fetch('Node.Results.demand_total', fill_value=float('nan'))['short vs base']
    assert unfilled.loc['2024-04-01'].isna().all().all()


def test_model_flag_of_all_scenarios(study):
    lines = study.scen.fetch('Line.Model')
    assert set(lines.columns.get_level_values('dataset')) == set(study.scenario_names)
    for scenario in study.scenario_names:
        assert lines[scenario]['node_from'].tolist() == ['N1', 'N2', 'N1', 'N3', 'N2']
        assert lines[scenario]['node_to'].tolist() == ['N2', 'N3', 'N3', 'N4', 'N4']
