import numpy as np
import pandas as pd
import pytest

from mesqual.testing import DummyPeriodSource, DummyScenario, DummySource

CONSTANT = DummyScenario(profile='constant', tz=None, start='2024-01-01', days=1, demand_factor=1.5)


@pytest.mark.parametrize('table', ['node_price', 'demand_load', 'generator_generation', 'line_flow'])
def test_same_spec_gives_same_values_and_another_seed_other_values(table):
    scenario = DummyScenario(seed=1)
    pd.testing.assert_frame_equal(DummySource(scenario).read(table), DummySource(scenario).read(table))
    other = DummySource(scenario.replace(seed=2)).read(table)
    assert not other.equals(DummySource(scenario).read(table))


def test_constant_profile_gives_hand_computable_values():
    source = DummySource(CONSTANT)
    network = source.network
    nodes, lines, generators = network.nodes, network.lines, network.generators
    price, demand = source.read('node_price'), source.read('demand_load')
    assert (price == nodes['base_price']).all().all()
    assert (demand == 1.5 * network.demands['base_demand']).all().all()
    assert (source.read('generator_generation') == 0.5 * generators['capacity']).all().all()
    assert (source.read('line_flow') == 0.5 * lines['capacity']).all().all()
    assert (source.read('line_capacity') == lines['capacity']).all().all()
    assert len(price) == 24


def test_generation_is_energy_per_period():
    source = DummySource(CONSTANT.replace(resolution='15min'))
    generation = source.read('generator_generation')
    assert len(generation) == 96
    assert generation.sum().tolist() == pytest.approx((0.5 * source.network.generators['capacity'] * 24).tolist())


@pytest.mark.parametrize('day, steps', [('2024-03-31', 23), ('2024-10-27', 25), ('2024-06-01', 24)])
def test_dst_days_have_23_or_25_hours(day, steps):
    prices = DummySource(DummyScenario(start=day, days=1, tz='Europe/Berlin')).read('node_price')
    assert len(prices) == steps


def test_resolution_change_switches_the_time_step_from_its_date():
    scenario = DummyScenario(start='2024-06-01', days=2, resolution_change=('2024-06-02', '15min'))
    index = DummySource(scenario).read('node_price').index
    assert len(index) == 24 + 96
    steps = pd.Series(index[1:] - index[:-1])
    assert set(steps.iloc[:23]) == {pd.Timedelta('60min')}
    assert set(steps.iloc[24:]) == {pd.Timedelta('15min')}


def test_nan_gaps_missing_rows_and_duplicate_rows():
    scenario = CONSTANT.replace(
        nan_gaps=(('2024-01-01 02:00', '2024-01-01 04:00'),),
        missing_rows=(('2024-01-01 10:00', '2024-01-01 13:00'),),
        duplicate_rows=2,
    )
    prices = DummySource(scenario).read('node_price')
    assert len(prices) == 24 - 3 + 2
    assert prices.index.duplicated().sum() == 2
    assert prices.isna().all(axis=1).sum() == 2
    assert pd.Timestamp('2024-01-01 11:00') not in prices.index


def test_missing_tables_leave_the_inventory_and_reading_them_raises():
    source = DummySource(DummyScenario(missing_tables=('demand_load',)))
    assert 'demand_load' not in source.inventory.variables
    assert not source.has_table('demand_load')
    with pytest.raises(KeyError):
        source.read('demand_load')


def test_unknown_missing_table_raises():
    with pytest.raises(ValueError):
        DummySource(DummyScenario(missing_tables=('node_prices',)))


def test_fingerprint_follows_the_spec():
    base = DummyScenario()
    assert DummySource(base).inventory.fingerprint == DummySource(DummyScenario()).inventory.fingerprint
    assert DummySource(base).inventory.fingerprint != DummySource(base.replace(seed=5)).inventory.fingerprint


def test_period_source_numbers_runs_and_periods_per_day():
    scenario = DummyScenario(start='2024-03-30', days=2, tz='Europe/Berlin', resolution_change=('2024-03-31', '15min'))
    index = DummyPeriodSource(scenario).read('node_price').index
    assert not isinstance(index, pd.MultiIndex)
    assert index.name == 'time'
    assert index[0] == (20240330, 1) and index[23] == (20240330, 24)
    assert index[24] == (20240331, 1) and index[-1] == (20240331, 92)
    values = DummyPeriodSource(scenario).read('node_price').to_numpy()
    assert np.array_equal(values, DummySource(scenario).read('node_price').to_numpy())


def test_window_with_offset_selects_one_hour_of_a_fall_back_day():
    scenario = DummyScenario(
        start='2024-10-27', days=1, tz='Europe/Berlin', nan_gaps=(('2024-10-27 02:00+01:00', '2024-10-27 03:00+01:00'),)
    )
    prices = DummySource(scenario).read('node_price')
    nan_rows = prices.index[prices.isna().all(axis=1)]
    assert len(prices) == 25
    assert [t.isoformat() for t in nan_rows] == ['2024-10-27T02:00:00+01:00']


@pytest.mark.parametrize(
    'changed',
    [
        {'removed_objects': ('N2', 'G3', 'D1')},
        {'n_nodes': 5, 'n_lines': 7, 'n_generators': 6, 'n_demands': 10},
        {'text_changes': (('generators', 'G1', 'fuel', 'gas'),)},
    ],
)
def test_added_removed_or_changed_objects_leave_the_series_of_the_others_unchanged(changed):
    base, other = DummySource(DummyScenario()), DummySource(DummyScenario(**changed))
    for table in DummySource.TIME_SERIES_TABLES:
        base_df, other_df = base.read(table), other.read(table)
        changed_objects = [change[1] for change in other.scenario.text_changes]
        common = base_df.columns.intersection(other_df.columns).drop(changed_objects, errors='ignore')
        assert len(common) >= 2
        pd.testing.assert_frame_equal(other_df[common], base_df[common])


def _two_weeks(start: str) -> DummySource:
    return DummySource(DummyScenario(start=start, days=14, seed=3))


def _solar(source: DummySource) -> pd.Series:
    generators = source.network.generators
    return source.read('generator_generation')[generators.index[generators['fuel'] == 'solar']].sum(axis=1)


@pytest.mark.parametrize('start', ['2024-01-08', '2024-06-10'])
def test_solar_is_zero_at_night_and_peaks_near_noon(start):
    solar = _solar(_two_weeks(start))
    hourly_mean = solar.groupby(solar.index.hour).mean()
    assert (solar[(solar.index.hour < 4) | (solar.index.hour >= 22)] == 0).all()
    assert 11 <= hourly_mean.idxmax() <= 13
    assert hourly_mean[12] > 0


def test_solar_gives_more_energy_per_day_in_june_than_in_december():
    assert _solar(_two_weeks('2024-06-10')).sum() > 3 * _solar(_two_weeks('2024-12-09')).sum()


def test_demand_is_higher_in_the_evening_than_at_night_and_on_weekdays_than_on_weekends():
    demand = _two_weeks('2024-01-08').read('demand_load').sum(axis=1)
    hours = demand.index.hour
    assert demand[hours == 19].mean() > 1.2 * demand[hours == 3].mean()
    weekend = demand.index.day_of_week >= 5
    assert demand[~weekend].mean() > demand[weekend].mean()


def test_demand_is_higher_in_winter():
    assert (
        _two_weeks('2024-01-08').read('demand_load').mean().mean()
        > _two_weeks('2024-06-10').read('demand_load').mean().mean()
    )


def test_wind_has_no_daily_cycle_and_is_higher_in_winter():
    def wind(start):
        source = _two_weeks(start)
        generators = source.network.generators
        return source.read('generator_generation')[generators.index[generators['fuel'] == 'wind']].sum(axis=1)

    winter = wind('2024-01-08')
    hourly_mean = winter.groupby(winter.index.hour).mean()
    assert hourly_mean.max() < 1.3 * hourly_mean.min()
    assert winter.mean() > wind('2024-06-10').mean()


@pytest.mark.parametrize('start', ['2024-01-08', '2024-06-10'])
def test_prices_follow_net_demand(start):
    source = _two_weeks(start)
    generators = source.network.generators
    generation = source.read('generator_generation')
    renewable = generation[generators.index[generators['fuel'].isin(['solar', 'wind'])]].sum(axis=1)
    net_demand = source.read('demand_load').sum(axis=1) - renewable
    price = source.read('node_price').mean(axis=1)
    assert np.corrcoef(price, net_demand)[0, 1] > 0.5


def test_nuclear_is_flat_and_line_flows_stay_within_capacity():
    source = _two_weeks('2024-01-08')
    generators = source.network.generators
    nuclear = source.read('generator_generation')[generators.index[generators['fuel'] == 'nuclear']]
    capacity = generators.loc[nuclear.columns, 'capacity']
    assert ((nuclear / capacity) > 0.8).all().all()
    flow, line_capacity = source.read('line_flow'), source.read('line_capacity')
    assert (flow <= line_capacity).all().all()
    assert (-flow <= DummySource.DOWN_CAPACITY_FACTOR * line_capacity).all().all()


def test_values_at_15_minutes_match_the_values_at_60_minutes():
    hourly = DummySource(DummyScenario(start='2024-06-10', days=1))
    quarter_hourly = DummySource(DummyScenario(start='2024-06-10', days=1, resolution='15min'))
    for table, how in [('node_price', 'mean'), ('demand_load', 'mean'), ('generator_generation', 'sum')]:
        resampled = quarter_hourly.read(table).resample('60min').agg(how)
        relative_gap = (resampled - hourly.read(table)).abs().sum().sum() / hourly.read(table).abs().sum().sum()
        assert relative_gap < 0.05, table


def test_higher_demand_factor_raises_prices_and_gas_output():
    base, high = (
        _two_weeks('2024-01-08'),
        DummySource(DummyScenario(start='2024-01-08', days=14, seed=3, demand_factor=1.2)),
    )
    assert (high.read('node_price').mean() > base.read('node_price').mean()).all()
    assert high.read('generator_generation')['G3'].sum() > base.read('generator_generation')['G3'].sum()
    pd.testing.assert_series_equal(high.read('generator_generation')['G1'], base.read('generator_generation')['G1'])
