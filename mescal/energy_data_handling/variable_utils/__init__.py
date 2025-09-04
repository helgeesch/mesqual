"""
MESCAL Energy Data Variable Utilities

This package provides specialized utilities for processing and transforming energy system variables
in MESCAL (Modular Energy Scenario Comparison Analysis Library). These utilities handle common
operations on energy data including flow aggregations, price calculations, congestion analysis,
and bidirectional data processing.

Key Components:
- RegionalTradeBalanceCalculator: Aggregates bidirectional power flows between regions
- VolumeWeightedPriceAggregator: Computes volume-weighted electricity prices for regions  
- CongestionRentCalculator: Calculates congestion rents for transmission lines
- AggregatedColumnAppender: Aggregates columns by common identifiers (e.g., technology types)
- UpDownNetAppender: Processes bidirectional data to create net and total columns

These utilities are designed to work with MESCAL's multi-scenario, multi-regional energy data
structures and support both single-level and hierarchical DataFrame operations.
"""

from .regional_trade_balance_calculator import RegionalTradeBalanceCalculator, FlowType
from .volume_weighted_price_aggregator import VolumeWeightedPriceAggregator
from .congestion_rent import CongestionRentCalculator
from .aggregate_cols_with_part_in_common import AggregatedColumnAppender
from .aggregate_up_down_directions_to_net_column import UpDownNetAppender

__all__ = [
    'RegionalTradeBalanceCalculator',
    'FlowType',
    'VolumeWeightedPriceAggregator',
    'CongestionRentCalculator',
    'AggregatedColumnAppender',
    'UpDownNetAppender',
]