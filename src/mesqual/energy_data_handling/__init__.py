"""Tools for energy time series, network flows and areas.

The tools work on pandas time series and network models. A quantity is
intensive (prices, power) or extensive (energy, volumes); the conversion
between time granularities depends on this.

- Time series: `TimeSeriesGranularityAnalyzer` and
  `TimeSeriesGranularityConverter` find and convert the time granularity;
  `TimeSeriesGapHandler` marks gaps.
- Network flows: `NetworkLineFlowsData` holds the sent and received flows of
  lines in both directions.
- `area_accounting`: area and border models from nodes and lines, and area
  variables such as area prices and sums.
- `variable_utils`: trade balances, congestion rents, net and total columns
  from up and down columns, and sums of columns with a common part.
- `model_handling`: properties of related model objects as new columns, and
  combination columns of from/to pairs.
"""

from .granularity_analyzer import TimeSeriesGranularityAnalyzer, GranularityError
from .granularity_converter import TimeSeriesGranularityConverter
from .time_series_gap_handling import TimeSeriesGapHandler

from .network_lines_data import NetworkLineFlowsData

from . import area_accounting
from . import variable_utils
from . import model_handling

from .area_accounting import (
    AreaModelGenerator,
    AreaBorderModelGenerator,
    AreaBorderNamingConventions,
    AreaBorderGeometryCalculator,
    AreaPriceCalculator,
    AreaSumCalculator,
)

from .variable_utils import (
    RegionalTradeBalanceCalculator,
    LossMode,
    CongestionRentCalculator,
    AggregatedColumnAppender,
    UpDownNetAppender,
)

from .model_handling import (
    MembershipPropertyEnricher,
    DirectionalMembershipPropertyEnricher,
    StringMembershipPairsAppender,
    TupleMembershipPairsAppender,
)

__all__ = [
    # Core time series processing
    'TimeSeriesGranularityAnalyzer',
    'GranularityError',
    'TimeSeriesGranularityConverter',
    'TimeSeriesGapHandler',
    # Network flow data structures
    'NetworkLineFlowsData',
    # Subpackages
    'area_accounting',
    'variable_utils',
    'model_handling',
    # Key area accounting classes
    'AreaModelGenerator',
    'AreaBorderModelGenerator',
    'AreaBorderNamingConventions',
    'AreaBorderGeometryCalculator',
    'AreaPriceCalculator',
    'AreaSumCalculator',
    # Key variable utility classes
    'RegionalTradeBalanceCalculator',
    'LossMode',
    'CongestionRentCalculator',
    'AggregatedColumnAppender',
    'UpDownNetAppender',
    # Key model handling classes
    'MembershipPropertyEnricher',
    'DirectionalMembershipPropertyEnricher',
    'StringMembershipPairsAppender',
    'TupleMembershipPairsAppender',
]
