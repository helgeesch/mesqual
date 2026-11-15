"""Synthetic scenarios for tests and examples, without an external platform package.

The platform is also the reference implementation of a platform package: flag
specs with variants, parameters and memberships, a flag family, derived
flags, area and border layers, availability from a source inventory, and
default attributes.

What `DummyDataset` gives for the default `DummyScenario`:

- Model flags `Country.Model`, `Node.Model`, `Line.Model`,
  `Generator.Model` and `Demand.Model`, with points for the nodes and
  square polygons for the countries, so map code runs without geodata.
- Time series: node price, demand per product (`total`, `hourly`,
  `quarter_hourly`), both with the parameter `solution`, generation, line flow
  `Line.Results.flow_{side}_{direction}` and line capacity. The source has
  the `total` demand, the `sent` `net` flow and the `up` capacity. The other
  variants derive from them through `parent_dataset.fetch`, so an override
  of a source variant reaches them.
- Derived flags that list their inputs in `INPUT_FLAGS`: line loading,
  node demand, and generation per node and fuel (a flag family).
- The area layer `Country` and the border layer `CountryBorder`, with area
  price, demand and generation, and border flows, capacities and price
  spread.
- Availability: a spec reads one raw table (`SPEC_TABLE`) or derives from
  input flags (`INPUT_FLAGS`). `missing_tables` of a `DummyScenario`
  removes raw tables, so the flags that need them are unavailable.

`make_dummy_study()` builds a ready study with comparisons;
`examples/quickstart.py` runs on it.

Examples:
    >>> dataset = DummyDataset(DummyScenario(name='base', days=2))
    >>> dataset.fetch('Country.Results.price').columns.tolist()
    ['C1', 'C2']
"""

from mesqual.testing.dataset import (
    DummyConfig,
    DummyDataset,
    DummyPeriodDataset,
    make_dummy_dataset,
)
from mesqual.testing.network import DummyNetwork
from mesqual.testing.scenario import DummyScenario
from mesqual.testing.source import DummyPeriodSource, DummySource
from mesqual.testing.study import make_dummy_study

__all__ = [
    'DummyConfig',
    'DummyDataset',
    'DummyNetwork',
    'DummyPeriodDataset',
    'DummyPeriodSource',
    'DummyScenario',
    'DummySource',
    'make_dummy_dataset',
    'make_dummy_study',
]
