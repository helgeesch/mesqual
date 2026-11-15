import pandas as pd
import pytest

from mesqual.testing import DummyScenario


def test_replace_returns_changed_copy_and_keeps_original():
    base = DummyScenario(name='base', days=3)
    high = base.replace(name='high', demand_factor=1.2)
    assert (high.name, high.demand_factor, high.days) == ('high', 1.2, 3)
    assert (base.name, base.demand_factor) == ('base', 1.0)


def test_split_gives_consecutive_named_parts():
    first, second = DummyScenario(name='base', start='2024-03-30', days=3).split('2024-03-31')
    assert (first.name, first.start, first.days) == ('base.1', '2024-03-30', 1)
    assert (second.name, second.start, second.days) == ('base.2', '2024-03-31', 2)


@pytest.mark.parametrize('at', ['2024-03-30', '2024-04-02', '2024-01-01'])
def test_split_outside_the_horizon_raises(at):
    with pytest.raises(ValueError):
        DummyScenario(start='2024-03-30', days=3).split(at)


def test_unknown_profile_raises():
    with pytest.raises(ValueError):
        DummyScenario(profile='weekly')


def test_resolution_on_follows_the_resolution_change():
    scenario = DummyScenario(start='2024-03-30', days=3, resolution_change=('2024-03-31', '15min'))
    assert scenario.resolution_on(pd.Timestamp('2024-03-30')) == '60min'
    assert scenario.resolution_on(pd.Timestamp('2024-03-31')) == '15min'
    assert scenario.resolution_on(pd.Timestamp('2024-04-01')) == '15min'
