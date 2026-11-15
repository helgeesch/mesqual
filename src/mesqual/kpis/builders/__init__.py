"""KPI builders: create many KPI definitions in one chain of calls.

- `KPIBuilder`: the base class.
- `FlagAggKPIBuilder`: `FlagAggKPIDefinition` objects for combinations of
  flags, aggregations and objects.
- `ComparisonKPIBuilder`: `ComparisonKPIDefinition` objects from base
  definitions and comparisons.
"""

from mesqual.kpis.builders.base import KPIBuilder
from mesqual.kpis.builders.flag_agg_builder import FlagAggKPIBuilder
from mesqual.kpis.builders.comparison_builder import ComparisonKPIBuilder

__all__ = [
    'KPIBuilder',
    'FlagAggKPIBuilder',
    'ComparisonKPIBuilder',
]
