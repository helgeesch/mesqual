"""Energy Data Handling Module.

This comprehensive module provides utilities for handling energy-specific time series data operations,
network flow analysis, area-level aggregations, and multi-scenario energy system modeling within the
MESCAL (Modular Energy Scenario Comparison Analysis Library) framework.

The module is designed to work with pandas time series data and energy system networks, particularly
for energy market analysis where different data streams may have varying temporal granularities and
require specific handling based on whether they represent intensive (prices, power levels) or
extensive (volumes, energy amounts) quantities.

Core Components
---------------

**Time Series Processing:**
- Granularity analysis and conversion between different temporal resolutions
- Gap detection and handling in time series data
- Support for intensive vs. extensive quantity conversions

**Network Flow Analysis:**
- Bidirectional transmission line flow data structures
- Network capacity modeling with loss considerations
- Flow direction conventions for complex network topologies

**Area-Level Aggregations:**
- Node-to-area aggregation with geographic modeling
- Cross-border flow analysis and capacity calculations
- Price aggregation using volume-weighted methods
- Multi-level area hierarchies (countries, bidding zones, regions)

**Variable Utilities:**
- Regional trade balance calculations
- Volume-weighted price aggregations
- Congestion rent analysis
- Directional data processing (up/down, net flows)

**Model Handling:**
- DataFrame enrichment with model properties
- Membership-based property propagation
- Combination identifier creation for paired relationships

Key Features
------------
- **Multi-Granularity Support:** Handle hourly, quarter-hourly, and custom time resolutions
- **Network Topology:** Complete transmission network modeling with losses
- **Geographic Analysis:** Spatial analysis and area-based aggregations
- **Market Analysis:** Price convergence, arbitrage, and cross-border studies
- **Data Validation:** Comprehensive input validation and error handling
- **Performance Optimized:** Efficient algorithms for large-scale energy system analysis

Example Usage
-------------
```python
import pandas as pd
from mescal.energy_data_handling import (
    TimeSeriesGranularityAnalyzer,
    TimeSeriesGranularityConverter,
    NetworkLineFlowsData,
    AreaModelGenerator,
    RegionalTradeBalanceCalculator
)

# Analyze time series granularity
analyzer = TimeSeriesGranularityAnalyzer()
granularity = analyzer.get_granularity_as_timedelta(datetime_index)

# Convert between granularities
converter = TimeSeriesGranularityConverter()
hourly_data = converter.convert_to_target_granularity(
    quarter_hourly_data, target_granularity_minutes=60
)

# Handle network flows with losses
flows = NetworkLineFlowsData(sent_up, received_up, sent_down, received_down)

# Aggregate to area level
area_generator = AreaModelGenerator(nodes_df, area_column='country')
area_model = area_generator.generate_area_model()
```

This module forms the foundation of MESCAL's energy system analysis capabilities, providing
comprehensive tools for multi-scenario, multi-regional energy market modeling and analysis.
"""

# Core time series processing
from .granularity_analyzer import TimeSeriesGranularityAnalyzer, GranularityError
from .granularity_converter import (
    TimeSeriesGranularityConverter, 
    GranularityConversionError, 
    SamplingMethodEnum
)
from .time_series_gap_handling import TimeSeriesGapHandler

# Network flow data structures
from .network_lines_data import NetworkLineFlowsData, NetworkLineCapacitiesData

# Import all subpackages for convenient access
from . import area_accounting
from . import variable_utils  
from . import model_handling

# Import key classes from subpackages for direct access
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
    VolumeWeightedPriceAggregator,
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
    'VolumeWeightedPriceAggregator',
    'CongestionRentCalculator',
    'AggregatedColumnAppender',
    'UpDownNetAppender',
    
    # Key model handling classes
    'MembershipPropertyEnricher',
    'DirectionalMembershipPropertyEnricher',
    'StringMembershipPairsAppender', 
    'TupleMembershipPairsAppender',
]