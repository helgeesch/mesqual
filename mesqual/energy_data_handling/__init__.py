"""Energy Data Handling Module.

This comprehensive module provides utilities for handling energy-specific time series data operations,
network flow analysis and area-level aggregations within the MESQUAL framework.

The module is designed to work with pandas time series data and energy system networks, particularly
for energy market analysis where different data streams may have varying temporal granularities and
require specific handling based on whether they represent intensive (prices, power levels) or
extensive (volumes, energy amounts) quantities.

Core Components
---------------

Time Series Processing:
    - Granularity analysis and conversion between different temporal resolutions
    - Gap detection and handling in time series data
    - Support for intensive vs. extensive quantity conversions

Network Flow Analysis:
    - Bidirectional transmission line flow data structures
    - Network capacity modeling with loss considerations
    - Flow direction conventions for complex network topologies

Area-Level Aggregations:
    - Node-to-area aggregation modules with geographic modeling
    - Cross-border flow analysis and capacity calculation modules
    - Price aggregation modules using volume-weighted methods

Variable Utilities:
    - Regional trade balance calculations
    - Volume-weighted price aggregations
    - Congestion rent analysis
    - Directional data processing (up/down, net flows)

Model Handling:
    - DataFrame enrichment with model properties
    - Membership-based property propagation
    - Combination identifier creation for paired relationships
"""

from .granularity_analyzer import TimeSeriesGranularityAnalyzer, GranularityError
from .granularity_converter import (
    TimeSeriesGranularityConverter, 
    GranularityConversionError, 
    SamplingMethodEnum
)
from .time_series_gap_handling import TimeSeriesGapHandler

from .network_lines_data import NetworkLineFlowsData, NetworkLineCapacitiesData

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
    BorderFlowCalculator,
    BorderPriceSpreadCalculator,
    BorderCapacityCalculator
)

from .variable_utils import (
    RegionalTradeBalanceCalculator,
    FlowType,
    CongestionRentCalculator,
    AggregatedColumnAppender,
    UpDownNetAppender
)

from .model_handling import (
    MembershipPropertyEnricher,
    DirectionalMembershipPropertyEnricher,
    StringMembershipPairsAppender,
    TupleMembershipPairsAppender
)

__all__ = [
    # Core time series processing
    'TimeSeriesGranularityAnalyzer',
    'GranularityError',
    'TimeSeriesGranularityConverter', 
    'GranularityConversionError',
    'SamplingMethodEnum',
    'TimeSeriesGapHandler',
    
    # Network flow data structures
    'NetworkLineFlowsData',
    'NetworkLineCapacitiesData',
    
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
    'BorderFlowCalculator',
    'BorderPriceSpreadCalculator',
    'BorderCapacityCalculator',
    
    # Key variable utility classes
    'RegionalTradeBalanceCalculator',
    'FlowType', 
    'CongestionRentCalculator',
    'AggregatedColumnAppender',
    'UpDownNetAppender',
    
    # Key model handling classes
    'MembershipPropertyEnricher',
    'DirectionalMembershipPropertyEnricher',
    'StringMembershipPairsAppender', 
    'TupleMembershipPairsAppender',
]

__version__ = '0.1.0'
