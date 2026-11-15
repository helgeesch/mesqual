"""KPI definitions: what a KPI computes.

- `KPIDefinition`: the base class.
- `FlagAggKPIDefinition`: aggregates the time series of a flag per object.
- `CustomKPIDefinition`: base class for study-specific logic.
- `ComparisonKPIDefinition`: compares the KPIs of two datasets.
"""

from mesqual.kpis.definitions.base import KPIDefinition
from mesqual.kpis.definitions.flag_aggregation import FlagAggKPIDefinition
from mesqual.kpis.definitions.custom import CustomKPIDefinition
from mesqual.kpis.definitions.comparison import ComparisonKPIDefinition

__all__ = [
    'KPIDefinition',
    'FlagAggKPIDefinition',
    'CustomKPIDefinition',
    'ComparisonKPIDefinition',
]
