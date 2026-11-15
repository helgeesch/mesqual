# Build a platform package

A platform package connects mesqual to one modeling tool or data source. It reads the output of the tool and gives one dataset per run, with flags that mesqual and the studies on the platform use. mesqual depends on no platform package; a platform package depends on mesqual.

A platform package has:

- a `SourceInventory` subclass: what the source of a run holds, built once per dataset from the file list and small reads;
- a `PlatformDataset` subclass, the platform dataset: one run of the tool. It builds the source inventory and holds the registry of interpreters;
- an interpreter base, a `Dataset` subclass, and the interpreters: each declares its flags with flag specs and reads them from the source or computes them from other flags;
- the area and border layers of the platform, with the interpreters of their flags.

This guide builds a small platform package for a toy tool that writes a folder of CSV files per run. `mesqual.testing` is the full reference implementation of a platform package.

## The source

Each run of the toy tool writes the countries, the nodes, the node prices and the node demand. Here, a function writes three runs: a base run, a run with higher prices, and a run without the demand file:

```python
from pathlib import Path

import pandas as pd


def write_run(folder: Path, price_factor: float = 1.0, with_demand: bool = True) -> Path:
    """Write the CSV files of one run of the toy tool."""
    folder.mkdir(exist_ok=True)
    time = pd.date_range('2024-01-01', periods=3, freq='h', name='time')
    countries = pd.DataFrame({'name': ['North', 'South']}, index=pd.Index(['C1', 'C2'], name='Country'))
    nodes = pd.DataFrame(
        {'voltage': [380, 220, 220], 'country': ['C1', 'C1', 'C2']}, index=pd.Index(['N1', 'N2', 'N3'], name='Node')
    )
    prices = pd.DataFrame({'N1': [40.0, 42.0, 45.0], 'N2': [50.0, 51.0, 49.0], 'N3': [60.0, 58.0, 61.0]}, index=time)
    demand = pd.DataFrame({'N1': [100.0, 120.0, 110.0], 'N2': [80.0, 90.0, 85.0], 'N3': [50.0, 55.0, 60.0]}, index=time)
    countries.to_csv(folder / 'countries.csv')
    nodes.to_csv(folder / 'nodes.csv')
    (prices * price_factor).to_csv(folder / 'prices.csv')
    if with_demand:
        demand.to_csv(folder / 'demand.csv')
    return folder


base_folder = write_run(Path('run_base'))
high_price_folder = write_run(Path('run_high_price'), price_factor=1.5)
no_demand_folder = write_run(Path('run_no_demand'), with_demand=False)
```

## The platform dataset

The platform dataset takes the folder of one run. It builds the source inventory in `build_source_inventory`, and `get_child_dataset_type()` gives the interpreter base:

```python
from dataclasses import dataclass

from mesqual.datasets import Availability, Dataset, FetchContext, FileRead, PlatformDataset, SourceInventory


@dataclass(frozen=True)
class CsvInventory(SourceInventory):
    """The CSV files of one run."""
    files: frozenset[str]


class CsvInterpreter(Dataset):
    """Base of the interpreters of CsvDataset."""


class CsvDataset(PlatformDataset):
    """One run of the toy tool: a folder of CSV files."""

    def __init__(self, folder: Path, **kwargs):
        self.folder = folder
        super().__init__(name=folder.name, **kwargs)

    def build_source_inventory(self) -> CsvInventory:
        files = sorted(path.name for path in self.folder.glob('*.csv'))
        fingerprint = repr([(file, FileRead.of(self.folder / file).key) for file in files])
        return CsvInventory(fingerprint=fingerprint, files=frozenset(files))

    @classmethod
    def get_child_dataset_type(cls) -> type[CsvInterpreter]:
        return CsvInterpreter
```

- The source inventory reads no time series. Build its `fingerprint` from `FileRead.of(path).key` per file.
- `PlatformDataset.__init__` runs the optional hook `_default_attributes`, which may read the source inventory. So set the fields that `build_source_inventory` reads, here `folder`, before `super().__init__`.
- The interpreters reach the platform dataset through `self.parent_dataset`. For an interpreter base with constructor arguments, see [`PlatformDataset.__init__`](../api_reference/datasets.md#mesqual.datasets.PlatformDataset.__init__).
- A study subclasses the interpreter base for its own interpreters ([Extend a platform in a study](extend-a-platform-in-a-study.md)). Give the base the helpers and the rules that every interpreter of the platform shares, and document them in its docstring.

## Flag specs

A flag spec declares one flag: name, description, unit, index, and how it relates to other flags. The specs are the only place that describes a flag:

```python
from mesqual.enums import TopologyTypeEnum
from mesqual.flag import ModelFlagSpec, TimeSeriesFlagSpec
from mesqual.units import Units

COUNTRY_MODEL_SPEC = ModelFlagSpec(
    name='Country.Model', description='Countries.', index='Country', topology_type=TopologyTypeEnum.Area,
)
NODE_MODEL_SPEC = ModelFlagSpec(
    name='Node.Model', description='Nodes of the network, each in a country.',
    index='Node', topology_type=TopologyTypeEnum.Node, memberships={'country': 'Country.Model'},
)
NODE_PRICE_SPEC = TimeSeriesFlagSpec(
    name='Node.Results.price', description='Price at the node.', unit=Units.EUR_per_MWh, model_flag='Node.Model',
)
NODE_DEMAND_SPEC = TimeSeriesFlagSpec(
    name='Node.Results.demand', description='Demand at the node.', unit=Units.MW, model_flag='Node.Model',
    fill_value=0,
)
NODE_DEMAND_COST_SPEC = TimeSeriesFlagSpec(
    name='Node.Results.demand_cost', description='Price times demand at the node: the cost of the demand per hour.',
    unit=Units.EUR, model_flag='Node.Model', fill_value=0,
)
```

- A model spec gives one row per object. Its `topology_type` says what the objects are, and `memberships` names the columns that refer to other model flags. The area layer below uses the membership `country`.
- A time series spec has one column per object of its `model_flag`. Its `fill_value` is the value for an object that one side of a comparison lacks.
- The flag names follow the grammar of the platform. mesqual reads no meaning from a name.

Variants (`flow_{direction}`), parameters (`solution='first'`) and flag families are in [Flags and layers](../architecture/flags-and-layers.md).

## Interpreters

An interpreter subclasses the interpreter base, declares its specs in `flag_specs` and implements two hooks:

- `_is_available(flag)`: whether the run has the flag, from cheap checks of the source inventory. It returns `True`, `False`, `None` for unknown (the default), or an `Availability` with a reason.
- `_fetch(flag, effective_config, **params)`: the frame of the flag. It records each source file it reads with `FetchContext.record_file_read`, so an edited file makes cached results stale.

`register_interpreter` registers the interpreter on the platform dataset class:

```python
@CsvDataset.register_interpreter
class CsvFileInterpreter(CsvInterpreter):
    """Flags that the tool writes, one CSV file each."""
    flag_specs = (COUNTRY_MODEL_SPEC, NODE_MODEL_SPEC, NODE_PRICE_SPEC, NODE_DEMAND_SPEC)
    FILE_OF_FLAG = {
        'Country.Model': 'countries.csv',
        'Node.Model': 'nodes.csv',
        'Node.Results.price': 'prices.csv',
        'Node.Results.demand': 'demand.csv',
    }

    def _is_available(self, flag: str) -> Availability:
        file = self.FILE_OF_FLAG[flag]
        found = file in self.parent_dataset.source_inventory.files
        return Availability(found, f'{file} {"in" if found else "not in"} the run folder')

    def _fetch(self, flag: str, effective_config, **params) -> pd.DataFrame:
        path = self.parent_dataset.folder / self.FILE_OF_FLAG[flag]
        FetchContext.record_file_read(path)
        if isinstance(self.flag_index.get_spec(flag), ModelFlagSpec):
            return pd.read_csv(path, index_col=0)
        return pd.read_csv(path, index_col=0, parse_dates=True).rename_axis(columns='Node')


dataset = CsvDataset(base_folder)
nodes = dataset.fetch('Node.Model')
prices = dataset.fetch('Node.Results.price')
print(prices.columns.tolist())
```

```text
['N1', 'N2', 'N3']
```

The platform dataset routes each fetch to the interpreter that declares the flag.

## Derived flags

A derived flag is computed from other flags. Its interpreter fetches each input through `self.parent_dataset` and passes `effective_config` on: `self.parent_dataset.fetch(flag, effective_config)`. So an override of an input reaches the derived flag, and the config of the call reaches the inputs ([the rule](../architecture/core-contracts.md#1-datasets)). Pass an input only the parameters that its spec declares; another parameter raises `ValueError`. `Availability.of_inputs` combines the availability of the inputs:

```python
@CsvDataset.register_interpreter
class CsvDemandCostInterpreter(CsvInterpreter):
    """The cost of the demand, from the node price and the node demand."""
    flag_specs = (NODE_DEMAND_COST_SPEC,)

    def _is_available(self, flag: str) -> Availability:
        inputs = ('Node.Results.price', 'Node.Results.demand')
        return Availability.of_inputs({f: self.parent_dataset.availability(f) for f in inputs})

    def _fetch(self, flag: str, effective_config, **params) -> pd.DataFrame:
        prices = self.parent_dataset.fetch('Node.Results.price', effective_config)
        demand = self.parent_dataset.fetch('Node.Results.demand', effective_config)
        return prices * demand
```

## Availability

The run without the demand file cannot answer the demand, nor the flags that need it. `describe_flags` hides unavailable specs by default; `available_only=False` shows them. `availability(flag).reason` says why:

```python
no_demand = CsvDataset(no_demand_folder)
flags = no_demand.describe_flags(available_only=False)
print(flags['availability'].to_dict())
print(no_demand.availability('Node.Results.demand_cost').reason)
```

```text
{'Country.Model': 'available', 'Node.Model': 'available', 'Node.Results.price': 'available', 'Node.Results.demand': 'unavailable', 'Node.Results.demand_cost': 'unavailable'}
CsvDemandCostInterpreter._is_available: unavailable input Node.Results.demand (CsvFileInterpreter._is_available: demand.csv not in the run folder)
```

A fetch of an unavailable flag raises `FlagUnavailableError` with the reason. A collection of runs leaves out the runs on which a flag is unavailable.

## Add an area layer

An area layer groups the objects of a model flag into areas through a membership column. The platform registers its layers and declares a layer spec for each flag that it aggregates: an `AreaFlagSpec` names the base spec and the aggregation. An `AreaFlagInterpreter` answers the layer specs:

```python
from mesqual.flag import AreaFlagSpec, AreaLayer
from mesqual.layer_flags import AreaFlagInterpreter, Sum, WeightedMean

CsvDataset.register_area_layer(AreaLayer(
    name='Country', index='Country', base_model='Node.Model', layer_model='Country.Model',
    description='Countries, each a group of nodes through the node column country.',
))


@CsvDataset.register_interpreter
class CsvAreaFlags(AreaFlagInterpreter):
    """Flags on the area layers of CsvDataset."""
    flag_specs = (
        AreaFlagSpec(
            name='{area_layer}.Results.price', base=NODE_PRICE_SPEC,
            aggregation=WeightedMean(weight='Node.Results.demand'),
            description='Mean of the node prices of the area, weighted by the node demand.',
        ),
        AreaFlagSpec(
            name='{area_layer}.Results.demand_cost', base=NODE_DEMAND_COST_SPEC, aggregation=Sum(),
            description='Sum of the cost of the demand of the nodes of the area.',
        ),
    )


dataset = CsvDataset(base_folder)
country_prices = dataset.fetch('Country.Results.price')
country_demand_cost = dataset.fetch('Country.Results.demand_cost')
print(country_prices.round(1))
```

```text
Country                C1    C2
time
2024-01-01 00:00:00  44.4  60.0
2024-01-01 01:00:00  45.9  58.0
2024-01-01 02:00:00  46.7  61.0
```

The price of `C1` is the mean of the prices of `N1` and `N2`, weighted by their demand.

The layer spec has a template name: `{area_layer}` stands for each area layer, so a study that adds a layer gets its flags too. A layer flag is available when the flags its fetch reads are available. Border layers (`BorderLayer`, `BorderFlagSpec`) work the same way: see [Flags and layers](../architecture/flags-and-layers.md#5-layers).

## Use the package in a study

A study builds its scenarios from the platform dataset:

```python
from mesqual import StudyManager

study = StudyManager.factory_from_scenarios(
    scenarios=[CsvDataset(base_folder), CsvDataset(high_price_folder)],
    comparisons=[('run_high_price', 'run_base')],
)
country_price_deltas = study.comp.fetch('Country.Results.price')
```

From here on, the study works as in [Use mesqual in a study](use-in-a-study.md).

## Test the package

The smallest test of a platform package fetches every flag of every spec on a small run that has every source file. On a run without a file, `fetch` raises `FlagUnavailableError` for the flags that need it. A full test also fetches each parameter value of a flag (see `flag_info(flag).params`); the loop below passes no parameters:

```python
def fetch_every_flag(dataset: PlatformDataset) -> None:
    """Fetch every flag of every spec of the dataset."""
    for spec in dataset.flag_index.specs:
        for flag in spec.expand(dataset):
            dataset.fetch(flag)


fetch_every_flag(CsvDataset(base_folder))
```

`mesqual.testing` shows more of a platform package: a config class, default attributes (`_default_attributes`), variants and parameters, a flag family, and a border layer.
