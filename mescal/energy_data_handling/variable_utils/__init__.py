"""
MESCAL Energy Data Variable Utilities

This package provides specialized utilities for processing and transforming energy system variables
in MESCAL (Modular Energy Scenario Comparison Analysis Library). These utilities handle common
operations on energy data including flow aggregations, price calculations, congestion analysis,
and bidirectional data processing.

Key Components:
    - RegionalTradeBalanceCalculator: Aggregates bidirectional power flows between regions
    - CongestionRentCalculator: Calculates congestion rents for transmission lines
    - AggregatedColumnAppender: Aggregates columns by common identifiers (e.g., technology types)
    - UpDownNetAppender: Processes bidirectional data to create net and total columns
"""

from .regional_trade_balance_calculator import RegionalTradeBalanceCalculator, FlowType
from .congestion_rent import CongestionRentCalculator
from .aggregate_cols_with_part_in_common import AggregatedColumnAppender
from .aggregate_up_down_directions_to_net_column import UpDownNetAppender

__all__ = [
    'RegionalTradeBalanceCalculator',
    'FlowType',
    'CongestionRentCalculator',
    'AggregatedColumnAppender',
    'UpDownNetAppender',
]

__version__ = '0.1.0'

