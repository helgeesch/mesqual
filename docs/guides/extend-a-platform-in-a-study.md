# Extend a platform in a study

A study often needs more than its platform package gives: its own model data, a changed platform flag, its own areas, its own variables. A study adds them on a subclass of the platform dataset. The registrations apply to this class and its subclasses only, so the platform package and other studies stay unchanged.

The examples build on the synthetic platform `mesqual.testing`. They add regions, each a group of countries, and the net position of each node. The steps are the same on every platform.

## The study dataset and the interpreter base

The study dataset is a subclass of the platform dataset. The study builds all its scenarios from it:

```python
from typing import ClassVar

import pandas as pd
import shapely

from mesqual.enums import TopologyTypeEnum
from mesqual.flag import ModelFlagSpec, TimeSeriesFlagSpec
from mesqual.testing import DummyDataset
from mesqual.testing.dataset import DummyInterpreter, DummyModelInterpreter
from mesqual.units import Units

REGION_OF_COUNTRY = {'C1': 'North', 'C2': 'North', 'C3': 'South'}


class StudyDataset(DummyDataset):
    """The synthetic platform with the regions of the study."""


print(StudyDataset.get_child_dataset_type().__name__)
```

```text
DummyInterpreter
```

An interpreter answers a set of flags. A study interpreter subclasses the interpreter base of the platform, `StudyDataset.get_child_dataset_type()`. The base gets its constructor arguments from the platform dataset, and its docstring gives the rules of the platform for availability. On the synthetic platform, `DummyInterpreter` takes the availability of a spec from the source table in `SPEC_TABLE` or from the flags in `INPUT_FLAGS` that the spec fetches.

An interpreter declares its flags with flag specs in `flag_specs` and computes them in `_fetch(flag, effective_config, **params)`. It fetches its inputs through `self.parent_dataset` and passes `effective_config` to each fetch: `self.parent_dataset.fetch(flag, effective_config)`. So an override of an input reaches the interpreter, and the config of the call reaches the inputs ([the rule](../architecture/core-contracts.md#1-datasets)). Pass an input only the parameters that its spec declares; another parameter raises `ValueError`.

## Add model data

A model flag gives one row per object. Here, the regions come from the countries, so the interpreter fetches `Country.Model` and groups its polygons:

```python
@StudyDataset.register_interpreter
class RegionModelInterpreter(DummyInterpreter):
    """Regions, each a group of countries."""
    flag_specs = (
        ModelFlagSpec(
            name='Region.Model', description='Regions, each a group of countries.',
            index='Region', topology_type=TopologyTypeEnum.Area,
        ),
    )
    INPUT_FLAGS: ClassVar[dict[str, tuple[str, ...]]] = {'Region.Model': ('Country.Model',)}

    def _fetch(self, flag: str, effective_config, **params) -> pd.DataFrame:
        countries = self.parent_dataset.fetch('Country.Model', effective_config)
        regions = countries.index.map(REGION_OF_COUNTRY).rename('Region')
        return countries[['geometry']].groupby(regions).agg(shapely.union_all)
```

An interpreter that reads a file of the study records it with `FetchContext.record_file_read`, so the cache sees changes. Its availability follows the docstring of the interpreter base.

## Override a platform interpreter

An override replaces a whole interpreter on the dataset class: `register_interpreter(Override, override_interpreter=Original, version=1)`. `StudyDataset.get_answering_interpreter(flag)` gives the interpreter that answers the flag now: the one to override.

- The override answers its own `flag_specs`. A subclass of the original inherits its `flag_specs` tuple. To change one spec, build the tuple again with the new spec in place of the old one.
- Reuse the platform code with `super()._fetch(...)`.
- Increase `version` when the logic of the override changes, so cached results become stale.

Here, the override adds the membership column `region` to `Node.Model`:

```python
NODE_MODEL_SPEC = ModelFlagSpec(
    name='Node.Model', description='Nodes of the network, each with a point, a country and a region.',
    index='Node', topology_type=TopologyTypeEnum.Node,
    memberships={'country': 'Country.Model', 'region': 'Region.Model'},
)


@StudyDataset.register_interpreter(override_interpreter=DummyModelInterpreter, version=1)
class NodeModelWithRegion(DummyModelInterpreter):
    """Platform model tables; the nodes also have their region."""
    flag_specs = tuple(
        NODE_MODEL_SPEC if spec.name == 'Node.Model' else spec for spec in DummyModelInterpreter.flag_specs
    )

    def _fetch(self, flag: str, effective_config, **params) -> pd.DataFrame:
        df = super()._fetch(flag, effective_config, **params)
        if flag == 'Node.Model':
            df['region'] = df['country'].map(REGION_OF_COUNTRY)
        return df


print(StudyDataset.get_answering_interpreter('Node.Model').__name__)
```

```text
NodeModelWithRegion
```

The platform flags that fetch `Node.Model` through the parent dataset now get the table with the column `region`.

## Add an area layer

An area layer groups the objects of a model flag into areas through a membership column. The layer specs of the platform then give the area flags, for example `Region.Results.price`, the price weighted by node demand:

```python
from mesqual.flag import AreaLayer

StudyDataset.register_area_layer(AreaLayer(
    name='Region', index='Region', base_model='Node.Model', layer_model='Region.Model',
    description='Regions, each a group of nodes through the node column region.',
))
```

The layer needs a membership of the base model to the layer model (`memberships` of `Node.Model`), and a column of that name. The override above adds both. The layer model can also be an existing model flag, for example `Country.Model`.

## Compute a custom variable

A custom variable is a flag of the study, computed from other flags. Here, the net position of a node is its generation minus its demand:

```python
@StudyDataset.register_interpreter
class NodeNetPositionInterpreter(DummyInterpreter):
    """Generation minus demand per node."""
    flag_specs = (
        TimeSeriesFlagSpec(
            name='Node.Results.net_position', description='Generation minus demand at the node.',
            unit=Units.MW, model_flag='Node.Model', fill_value=0,
        ),
    )
    INPUT_FLAGS: ClassVar[dict[str, tuple[str, ...]]] = {
        'Node.Results.net_position': ('Generator.Results.generation', 'Generator.Model', 'Node.Results.demand_total'),
    }

    def _fetch(self, flag: str, effective_config, **params) -> pd.DataFrame:
        generation = self.parent_dataset.fetch('Generator.Results.generation', effective_config)
        node_of_generator = self.parent_dataset.fetch('Generator.Model', effective_config)['node']
        demand = self.parent_dataset.fetch('Node.Results.demand_total', effective_config)
        generation_per_node = generation.T.groupby(node_of_generator).sum().T
        return generation_per_node.reindex(columns=demand.columns, fill_value=0) - demand
```

The area layers of the platform do not cover the new flag: an area flag needs a layer spec on its base spec. To get the net position per country or region, register an `AreaFlagInterpreter` with an `AreaFlagSpec` on the new spec (see [Build a platform package](build-a-platform-package.md#add-an-area-layer)).

## Check the new flags

The describe methods show the new flags with their availability:

```python
from mesqual.testing import DummyScenario

base = DummyScenario(name='base', start='2024-03-28', days=3, n_countries=3)
dataset = StudyDataset(base)
region_flags = dataset.describe_flags(contains='Region')
net_position_info = dataset.flag_info('Node.Results.net_position')
new_flags = ('Region.Model', 'Region.Results.price', 'Node.Results.net_position')
print({flag: dataset.availability(flag).label for flag in new_flags})
```

```text
{'Region.Model': 'available', 'Region.Results.price': 'available', 'Node.Results.net_position': 'available'}
```

Because the custom variable passes `effective_config` to its inputs, it follows the config of the call. With the first solution of the run, the demand input changes, and the net position changes with it:

```python
first_solution = {'param_defaults': {'solution': 'first'}}
net_position = dataset.fetch('Node.Results.net_position')
net_position_first = dataset.fetch('Node.Results.net_position', config=first_solution)
print(net_position_first.equals(net_position))
```

```text
False
```

## Analyze the study

The study builds all scenarios from `StudyDataset`:

```python
from mesqual import StudyManager

study = StudyManager.factory_from_scenarios(
    scenarios=[
        StudyDataset(base),
        StudyDataset(base.replace(name='high_demand', demand_factor=1.2)),
    ],
    comparisons=[('high_demand', 'base')],
)

regions = study.scen.get_dataset('base').fetch('Region.Model')        # North holds C1 and C2
region_prices = study.scen.fetch('Region.Results.price')              # area layer of the study
net_position_deltas = study.comp.fetch('Node.Results.net_position')   # custom variable
nodes = study.scen.get_dataset('base').fetch('Node.Model')            # has the column region
print(region_prices.columns.tolist())
```

```text
[('base', 'North'), ('base', 'South'), ('high_demand', 'North'), ('high_demand', 'South')]
```

KPIs, maps and dashboards work on the new flags as on every other flag: see [Use mesqual in a study](use-in-a-study.md#compute-kpis).

## In a notebook

A registration changes the class. A cell that runs twice registers new interpreter classes of the same name on one class, which raises `ValueError` or `AmbiguousOverrideError`. So in a notebook, define the study dataset class in the same cell as its registrations, and build the study again after each run of the cell.
