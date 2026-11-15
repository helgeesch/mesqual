import pandas as pd
import pytest

from mesqual import StudyManager
from mesqual.datasets import DatasetMergeCollection
from mesqual.testing import DummyDataset, DummyScenario, make_dummy_study


@pytest.fixture(scope='module')
def study() -> StudyManager:
    return make_dummy_study()


def test_study_has_four_scenarios_and_comparisons_against_base(study):
    assert study.scenario_names == ['base', 'high_demand', 'without_N4', 'short']
    assert study.comparison_names == ['high_demand vs base', 'without_N4 vs base', 'short vs base']


def test_base_is_merged_from_two_periods_with_a_resolution_change(study):
    base = study.scen.get_dataset('base')
    assert isinstance(base, DatasetMergeCollection)
    unsplit = DummyDataset(
        DummyScenario(name='base', start='2024-03-28', days=5, resolution_change=('2024-03-30', '15min'))
    )
    prices = base.fetch('Node.Results.price')
    pd.testing.assert_frame_equal(prices, unsplit.fetch('Node.Results.price'))
    steps = pd.Series(prices.index[1:] - prices.index[:-1]).unique()
    assert set(steps) == {pd.Timedelta('60min'), pd.Timedelta('15min')}


def test_scenarios_have_attributes(study):
    attributes = study.scen.get_attributes_concat_df()
    assert attributes.loc['base', 'periods'] == 2
    assert attributes.loc['high_demand', 'demand_factor'] == 1.2
    assert attributes.loc['short', 'days'] == 4
    assert {'first_date', 'last_date'} <= set(attributes.columns)


def test_comparisons_show_the_variation(study):
    demand_deltas = study.comp.fetch('Demand.Results.demand_total')
    assert (demand_deltas.xs('high_demand vs base', level='dataset', axis=1) > 0).all().all()
    price_deltas = study.comp.fetch('Node.Results.price').xs('high_demand vs base', level='dataset', axis=1)
    assert price_deltas.mean().gt(0).all()


def test_removed_node_has_no_price_delta_and_its_demand_is_filled_with_zero(study):
    comparison = study.comp.get_dataset('without_N4 vs base')
    base = study.scen.get_dataset('base')
    price_deltas = comparison.fetch('Node.Results.price')
    assert price_deltas['N4'].isna().all()
    assert price_deltas[['N1', 'N2', 'N3']].notna().all().all()
    demand_deltas = comparison.fetch('Node.Results.demand_total')
    pd.testing.assert_series_equal(demand_deltas['N4'], -base.fetch('Node.Results.demand_total')['N4'])


def test_less_than_two_days_raises():
    with pytest.raises(ValueError, match='at least 2 days'):
        make_dummy_study(days=1)
