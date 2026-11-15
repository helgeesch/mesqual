"""Ready-made study of synthetic scenarios."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from mesqual.study_manager import StudyManager
from mesqual.testing.dataset import DummyDataset, make_dummy_dataset
from mesqual.testing.scenario import DummyScenario


def make_dummy_study(
    start: str = '2024-03-28',
    days: int = 5,
    tz: str | None = 'Europe/Berlin',
    seed: int = 0,
    study_folder: str | Path | None = None,
) -> StudyManager:
    """Build a study of four synthetic scenarios with comparisons against 'base'.

    Scenarios:

    - 'base': split into two consecutive periods at the middle day, one
      `DummyDataset` per period, merged into one dataset. The series switch
      from 60 to 15 minutes at the same day (`resolution_change`).
    - 'high_demand': 'base' with 20% more demand, one dataset.
    - 'without_N4': 'base' without node N4 and its lines, generators and demands.
    - 'short': 'base' one day shorter.

    Comparisons: each other scenario against 'base', named '<scenario> vs base'
    by `DatasetComparison`.

    Every scenario has the attributes 'first_date', 'last_date', 'days',
    'demand_factor' and 'periods' (number of merged period datasets).

    The default horizon from Thursday 2024-03-28 has a weekend and the DST
    change on 2024-03-31 in Europe/Berlin.

    Args:
        start: First day.
        days: Number of days, at least 2.
        tz: Time zone of the time index, or None for naive timestamps.
        seed: Seed of the random noise.
        study_folder: Optional folder of the study, passed to `StudyManager`.

    Returns:
        The study, with the scenarios in `study.scen` and the comparisons in `study.comp`.

    Raises:
        ValueError: If `days` is less than 2.

    Examples:
        >>> study = make_dummy_study()
        >>> deltas = study.comp.fetch('Node.Results.price')
    """
    if days < 2:
        raise ValueError(f'make_dummy_study needs at least 2 days, got {days}.')
    split_day = (pd.Timestamp(start) + pd.Timedelta(days=days // 2)).strftime('%Y-%m-%d')
    base = DummyScenario(
        name='base',
        start=start,
        days=days,
        tz=tz,
        seed=seed,
        resolution='60min',
        resolution_change=(split_day, '15min'),
    )
    variations = [
        base.replace(name='high_demand', demand_factor=1.2),
        base.replace(name='without_N4', removed_objects=('N4',)),
        base.replace(name='short', days=days - 1),
    ]
    scenarios = [make_dummy_dataset(base.split(split_day))] + [DummyDataset(v) for v in variations]
    for dataset, scenario in zip(scenarios, [base, *variations]):
        dataset.set_attributes(demand_factor=scenario.demand_factor, periods=2 if dataset.name == 'base' else 1)
    return StudyManager.factory_from_scenarios(
        scenarios=scenarios,
        comparisons=[(v.name, base.name) for v in variations],
        study_folder=study_folder,
    )
