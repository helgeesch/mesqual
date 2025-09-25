"""
Model handling utilities for MESCAL energy data processing.

This package provides tools for enriching energy system DataFrames with properties
from related model objects and creating combination columns for paired relationships.

Key Components:
    - Membership property enrichers for adding related object properties to DataFrames
    - Directional relationship handling for from/to column pairs in network data
    - Membership pairs appenders for creating combination identifiers from paired columns

Example use cases:
    - Enriching generator data with node properties
      (e.g. generator_model_df then has node_voltage, node_country, ...)
    - Enriching line_model_df with node_from and node_to characteristics
      (e.g. region_from, region_to, voltage_from, voltage_to, ...)
"""

from .membership_property_enrichers import (
    MembershipTagging,
    MembershipPropertyEnricher,
    DirectionalMembershipPropertyEnricher,
)

from .membership_pairs_appender import (
    BaseMembershipPairsAppender,
    StringMembershipPairsAppender,
    TupleMembershipPairsAppender,
)

__all__ = [
    # Enums
    'MembershipTagging',
    
    # Property enrichers
    'MembershipPropertyEnricher',
    'DirectionalMembershipPropertyEnricher',
    
    # Pairs appenders
    'BaseMembershipPairsAppender',
    'StringMembershipPairsAppender',
    'TupleMembershipPairsAppender',
]

__version__ = '0.1.0'
