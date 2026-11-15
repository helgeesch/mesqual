"""KPIs: one value per object and dataset, with the attributes of both.

A KPI keeps its links to the dataset and to the model object. So a KPI
collection can filter KPIs by model properties, find related KPIs, convert
the units of mixed KPIs, and give the values to the map visualizations.
For a quick metric, a fetch and pandas are enough.

Components:

- `KPIBuilder`: builds a set of KPI definitions, for example
  `FlagAggKPIBuilder` and `ComparisonKPIBuilder`.
- `KPIDefinition`: what to compute. `FlagAggKPIDefinition` aggregates a
  flag, `CustomKPIDefinition` holds study-specific logic and
  `ComparisonKPIDefinition` compares the KPIs of two scenarios.
- `KPI`: a computed value with its attributes (`KPIAttributes`).
- `KPICollection`: filters, groups and exports KPIs.
- `Aggregations`, `ValueComparisons` and `ArithmeticValueOperations`: the
  operations of the definitions.

A custom metric can also be a flag of a study interpreter, aggregated
with `FlagAggKPIDefinition`. The metric is then also available as a time
series.

Examples:
    A quick metric with pandas:

    >>> study.scen.fetch('Country.Results.price').mean().unstack('dataset').round(2)
    dataset   base  high_demand  without_N4  short
    Country
    C1       42.10        53.46       42.10  39.97
    C2       46.13        58.71       41.14  43.48

    The same metric as KPIs:

    >>> from mesqual.kpis import Aggregations, FlagAggKPIBuilder
    >>> builder = FlagAggKPIBuilder().for_flag('Country.Results.price')
    >>> dataset.add_kpis_from_definitions(builder.with_aggregation(Aggregations.Mean).build())
    >>> kpis = dataset.kpi_collection.filter_by_model_properties(properties={'name': 'Country 1'})
    >>> kpis.to_dataframe()[['object_name', 'value']].round(2)
      object_name  value
    0          C1   42.1
"""

# Core classes
from mesqual.kpis.kpi import KPI
from mesqual.kpis.attributes import KPIAttributes
from mesqual.kpis.collection import KPICollection

# Definitions
from mesqual.kpis.definitions.base import KPIDefinition
from mesqual.kpis.definitions.flag_aggregation import FlagAggKPIDefinition
from mesqual.kpis.definitions.custom import CustomKPIDefinition
from mesqual.kpis.definitions.comparison import ComparisonKPIDefinition

# Aggregations
from mesqual.kpis.aggregations import (
    Aggregation,
    Aggregations,
    ValueComparison,
    ValueComparisons,
    ArithmeticValueOperation,
    ArithmeticValueOperations,
)

# Builders
from mesqual.kpis.builders.flag_agg_builder import FlagAggKPIBuilder
from mesqual.kpis.builders.comparison_builder import ComparisonKPIBuilder

__all__ = [
    # Core
    'KPI',
    'KPIAttributes',
    'KPICollection',
    # Definitions
    'KPIDefinition',
    'FlagAggKPIDefinition',
    'CustomKPIDefinition',
    'ComparisonKPIDefinition',
    # Aggregations
    'Aggregation',
    'Aggregations',
    'ValueComparison',
    'ValueComparisons',
    'ArithmeticValueOperation',
    'ArithmeticValueOperations',
    # Builders
    'FlagAggKPIBuilder',
    'ComparisonKPIBuilder',
]
